#include "wrftools/lcz.hpp"
#include "wrftools/error.hpp"
#include "wrftools/warp.hpp"

#include "third_party/nanoflann.hpp"

#include <netcdf.h>
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <cpl_conv.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <set>
#include <sstream>

namespace wrftools {
namespace {

double globalAttr(const NetcdfFile& file, const std::string& name) {
    const auto attribute = file.getAttribute("", name);
    if (attribute.numbers.empty()) throw UserError("geo_em file has no numeric global attribute: " + name);
    return attribute.numbers.front();
}

std::size_t dimensionLength(const NetcdfFile& file, const std::string& name) {
    for (const auto& dimension : file.dimensions())
        if (dimension.name == name) return dimension.length;
    throw UserError("geo_em file has no dimension: " + name);
}

// A great-circle-chord point cloud over a lat/lon pixel grid, for
// nanoflann - mirrors using_kdtree's own ECEF-ish projection (w2w.py:432-
// 453) exactly: R=6371 (km; the unit is arbitrary since only relative
// distances matter for a k-nearest query), no ellipsoid correction.
struct EcefPointCloud {
    std::vector<std::array<double, 3>> points;

    explicit EcefPointCloud(const std::vector<float>& lat, const std::vector<float>& lon) {
        constexpr double kEarthRadiusKm = 6371.0;
        points.reserve(lat.size());
        for (std::size_t i = 0; i < lat.size(); ++i) {
            const double phi = static_cast<double>(lat[i]) * std::numbers::pi / 180.0;
            const double theta = static_cast<double>(lon[i]) * std::numbers::pi / 180.0;
            points.push_back({kEarthRadiusKm * std::cos(phi) * std::cos(theta), kEarthRadiusKm * std::cos(phi) * std::sin(theta),
                kEarthRadiusKm * std::sin(phi)});
        }
    }

    [[nodiscard]] std::size_t kdtree_get_point_count() const { return points.size(); }
    [[nodiscard]] double kdtree_get_pt(std::size_t idx, std::size_t dim) const { return points[idx][dim]; }
    template <class Bbox>
    bool kdtree_get_bbox(Bbox&) const {
        return false;
    }
};

using PixelKdTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, EcefPointCloud>, EcefPointCloud, 3>;

// The indices of a pixel's `k` nearest pixels (by great-circle chord
// distance, ascending - including itself, always first at distance 0),
// matching using_kdtree's tree.query(..., k=kpoints) exactly. Queried on
// demand, one pixel at a time, rather than precomputed for the whole grid:
// only urban pixels are ever looked up, and with the default NPIX_AREA
// (NPIX_NLC^2 = 2025 neighbors) precomputing every pixel of a 350x350
// domain meant ~122k k=2025 queries plus ~1 GB of stored results - minutes
// of apparent hang for what is a handful of actual lookups.
class NeighborSearcher {
public:
    NeighborSearcher(const std::vector<float>& lat, const std::vector<float>& lon, std::size_t k)
        : cloud_(lat, lon), tree_(3, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10)), k_(k), distances_(k) {}

    [[nodiscard]] std::vector<std::uint32_t> query(std::size_t pixel) {
        std::vector<std::uint32_t> result(k_);
        tree_.knnSearch(cloud_.points[pixel].data(), static_cast<std::uint32_t>(k_), result.data(), distances_.data());
        return result;
    }

private:
    EcefPointCloud cloud_;
    PixelKdTree tree_;
    std::size_t k_;
    std::vector<double> distances_;
};

// pandas Series.mode()[0]'s tie-break: the SMALLEST value among those
// tied for the highest frequency (mode() itself returns every tied value
// in ascending order; [0] takes the first).
float modalValue(const std::vector<float>& values) {
    std::map<int, int> counts;
    for (float v : values) ++counts[static_cast<int>(std::lround(v))];
    int bestCount = -1, bestValue = 0;
    for (const auto& [value, count] : counts)  // std::map iterates in ascending key order
        if (count > bestCount) { bestCount = count; bestValue = value; }
    return static_cast<float>(bestValue);
}

std::string canonicalWgs84Wkt() {
    OGRSpatialReference srs;
    srs.importFromEPSG(4326);
    srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    char* wkt = nullptr;
    if (srs.exportToWkt(&wkt) != OGRERR_NONE || !wkt) throw UserError("Could not build the EPSG:4326 CRS.");
    std::string result(wkt);
    CPLFree(wkt);
    return result;
}

}  // namespace

WrfGridInfo wrfGridInfo(const NetcdfFile& file) {
    const int mapProj = static_cast<int>(globalAttr(file, "MAP_PROJ"));
    const double trueLat1 = globalAttr(file, "TRUELAT1");
    const double trueLat2 = globalAttr(file, "TRUELAT2");
    const double standLon = globalAttr(file, "STAND_LON");
    const double moadCenLat = globalAttr(file, "MOAD_CEN_LAT");
    const double cenLon = globalAttr(file, "CEN_LON");
    const double cenLat = globalAttr(file, "CEN_LAT");
    const double dx = globalAttr(file, "DX");
    const double dy = globalAttr(file, "DY");

    // Projection ids match w2w.py:581-660 exactly (same as
    // wrf_file.cpp's buildWrfCrs, MAP_PROJ==6 aside - see lcz.hpp's
    // comment on why that one case can't reuse Crs::lonLat()).
    Crs wrfCrs = [&]() -> Crs {
        switch (mapProj) {
            case 1: return Crs::lambert(trueLat1, trueLat2, {standLon, moadCenLat});
            case 2: return Crs::polar(trueLat1, standLon);
            case 3: return Crs::mercator(trueLat1, standLon);
            case 6: return Crs::fromProj4(std::format("+proj=eqc +lon_0={} +x_0=0 +y_0=0 +a=6370000 +b=6370000 +no_defs", standLon));
            default: throw UnsupportedError("w2w: unsupported MAP_PROJ " + std::to_string(mapProj));
        }
    }();

    // e, n: the domain's projected center, transformed from its own
    // CEN_LON/CEN_LAT through real WGS84 - see WrfGridInfo's doc comment
    // for why this deliberately doesn't go through Crs::toXy.
    const auto origin = Crs::wgs84().transformBbox({cenLon, cenLat, cenLon, cenLat}, wrfCrs);
    const double e = origin.minX, n = origin.minY;

    const double nx = static_cast<double>(dimensionLength(file, "west_east"));
    const double ny = static_cast<double>(dimensionLength(file, "south_north"));

    const double x0 = -(nx - 1.0) / 2.0 * dx + e;
    const double y0 = -(ny - 1.0) / 2.0 * dy + n;

    // Affine.translation(x0 - dx/2, y0 - dy/2) * Affine.scale(dx, dy),
    // written out as a GDAL-style [c, a, b, f, d, e] geotransform. dy is
    // POSITIVE (not GDAL's usual negative row height) - see WrfGridInfo's
    // doc comment.
    const std::array<double, 6> geotransform{x0 - dx / 2.0, dx, 0.0, y0 - dy / 2.0, 0.0, dy};

    return {wrfCrs, geotransform, static_cast<int>(nx), static_cast<int>(ny)};
}

void removeUrban(const std::filesystem::path& srcPath, const std::filesystem::path& dstPath, int npixNlc, std::optional<int> npixArea) {
    auto src = NetcdfFile::open(srcPath, NetcdfFile::Mode::ReadOnly);

    const int origNumLandCat = static_cast<int>(globalAttr(src, "NUM_LAND_CAT"));
    const int urbanCat = static_cast<int>(globalAttr(src, "ISURBAN"));
    const int waterCat = static_cast<int>(globalAttr(src, "ISWATER"));
    const bool hasLakeCat = src.hasAttribute("", "ISLAKE") && [&] {
        const int islake = static_cast<int>(globalAttr(src, "ISLAKE"));
        return islake > 0 && islake < origNumLandCat;
    }();
    const int lakeCat = hasLakeCat ? static_cast<int>(globalAttr(src, "ISLAKE")) : 0;

    auto luse = src.readFloat("LU_INDEX");
    auto luf = src.readFloat("LANDUSEF");
    auto greenf = src.readFloat("GREENFRAC");
    const auto lat = src.readFloat("XLAT_M");
    const auto lon = src.readFloat("XLONG_M");

    const auto luShape = src.shape("LU_INDEX");      // {1, ny, nx}
    const auto greenfShape = src.shape("GREENFRAC");  // {1, months, ny, nx}
    const std::size_t ny = luShape[1], nx = luShape[2];
    const std::size_t npix = ny * nx;
    const std::size_t months = greenfShape[1];

    // Pre-pass (add_wrf_version only, absent from w2w's `main`): collapse
    // any LCZ-range LU_INDEX values the SOURCE file already carries back
    // to ISURBAN, so re-running this against a file w2w itself already
    // produced is idempotent. Ranges match w2w.py exactly, including its
    // own slight over-inclusiveness (the LANDUSEF slice's upper bound is
    // one past the LU_INDEX threshold's - see lcz.hpp's removeUrban doc).
    if (origNumLandCat == 61 || origNumLandCat == 41) {
        const int lo = origNumLandCat == 61 ? 50 : 30;
        const int lufHiExclusive = origNumLandCat == 61 ? 61 : 41;
        for (std::size_t p = 0; p < npix; ++p)
            if (luse[p] > static_cast<float>(lo) && luse[p] < static_cast<float>(origNumLandCat)) luse[p] = static_cast<float>(urbanCat);
        for (std::size_t p = 0; p < npix; ++p) {
            double sum = 0.0;
            for (int cat = lo; cat < lufHiExclusive; ++cat) sum += luf[static_cast<std::size_t>(cat) * npix + p];
            luf[static_cast<std::size_t>(urbanCat - 1) * npix + p] = static_cast<float>(sum);
        }
        for (int cat = lo; cat < lufHiExclusive; ++cat)
            for (std::size_t p = 0; p < npix; ++p) luf[static_cast<std::size_t>(cat) * npix + p] = 0.0f;
    }

    std::vector<float> newluse = luse;
    std::vector<float> newluf = luf;
    std::vector<float> newgreenf = greenf;

    std::vector<bool> lufNatland(npix), lufUrb(npix);
    for (std::size_t p = 0; p < npix; ++p) {
        const bool natland = luf[static_cast<std::size_t>(urbanCat - 1) * npix + p] == 0.0f &&
                              luf[static_cast<std::size_t>(waterCat - 1) * npix + p] == 0.0f &&
                              (!hasLakeCat || luf[static_cast<std::size_t>(lakeCat - 1) * npix + p] == 0.0f);
        lufNatland[p] = natland;
        lufUrb[p] = luf[static_cast<std::size_t>(urbanCat - 1) * npix + p] != 0.0f;
    }

    const std::size_t effectiveNpixArea = npixArea.value_or(static_cast<std::size_t>(npixNlc) * static_cast<std::size_t>(npixNlc));
    if (effectiveNpixArea > npix)
        throw UserError("The area you selected is larger than the domain size: you chose an area of " + std::to_string(effectiveNpixArea) +
                         " pixels and the domain is " + std::to_string(npix) + " pixels. Reduce NPIX_AREA.");
    const std::size_t kpoints = std::min(npix, effectiveNpixArea);
    NeighborSearcher searcher(lat, lon, kpoints);

    std::vector<bool> luseUrb(npix), luseNatland(npix);
    for (std::size_t p = 0; p < npix; ++p) {
        luseUrb[p] = static_cast<int>(std::lround(luse[p])) == urbanCat;
        luseNatland[p] = static_cast<int>(std::lround(luse[p])) != urbanCat && static_cast<int>(std::lround(luse[p])) != waterCat &&
                          (!hasLakeCat || static_cast<int>(std::lround(luse[p])) != lakeCat);
    }

    for (std::size_t p = 0; p < npix; ++p) {
        if (!luseUrb[p]) continue;

        std::vector<std::uint32_t> auxKd;
        auxKd.reserve(static_cast<std::size_t>(npixNlc));
        for (std::uint32_t candidate : searcher.query(p)) {
            if (!luseNatland[candidate]) continue;
            auxKd.push_back(candidate);
            if (auxKd.size() == static_cast<std::size_t>(npixNlc)) break;
        }
        if (auxKd.empty())
            throw UserError("Not enough natural land points in the selected area: sampled " + std::to_string(npixNlc) + " pixels over an area of " +
                             std::to_string(effectiveNpixArea) + " - increase NPIX_AREA.");

        std::vector<float> auxLuse;
        auxLuse.reserve(auxKd.size());
        for (std::uint32_t idx : auxKd) auxLuse.push_back(luse[idx]);
        const float mkd = modalValue(auxLuse);

        std::vector<double> greenSum(months, 0.0);
        int greenCount = 0;
        for (std::uint32_t idx : auxKd) {
            if (static_cast<int>(std::lround(luse[idx])) != static_cast<int>(std::lround(mkd))) continue;
            for (std::size_t m = 0; m < months; ++m) greenSum[m] += greenf[m * npix + idx];
            ++greenCount;
        }
        newluse[p] = mkd;
        for (std::size_t m = 0; m < months; ++m) newgreenf[m * npix + p] = static_cast<float>(greenSum[m] / greenCount);
    }

    for (std::size_t p = 0; p < npix; ++p) {
        if (!lufUrb[p]) continue;
        const int destCat = static_cast<int>(std::lround(newluse[p]));
        newluf[static_cast<std::size_t>(destCat - 1) * npix + p] += luf[static_cast<std::size_t>(urbanCat - 1) * npix + p];
        newluf[static_cast<std::size_t>(urbanCat - 1) * npix + p] = 0.0f;
    }

    NetcdfFile::copyFile(srcPath, dstPath);
    auto dst = NetcdfFile::open(dstPath, NetcdfFile::Mode::ReadWrite);
    dst.writeFloat("LU_INDEX", newluse);
    dst.writeFloat("LANDUSEF", newluf);
    dst.writeFloat("GREENFRAC", newgreenf);

    NetcdfFile::Attribute numLandCat;
    numLandCat.name = "NUM_LAND_CAT";
    numLandCat.type = NC_INT;
    numLandCat.numbers = {21.0};
    dst.putAttribute("", numLandCat);
}

namespace {
std::string trimmed(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(trimmed(field));
    return fields;
}
}  // namespace

namespace {
std::map<int, UcpRow> parseUcpTable(std::istream& input, const std::string& sourceDescription) {
    std::string headerLine;
    if (!std::getline(input, headerLine)) throw UserError("UCP lookup table is empty: " + sourceDescription);
    const auto header = splitCsvLine(headerLine);
    std::map<std::string, std::size_t> columnIndex;
    for (std::size_t i = 1; i < header.size(); ++i) columnIndex[header[i]] = i;  // column 0 is the (unnamed) LCZ class index

    const auto require = [&](const char* name) {
        const auto found = columnIndex.find(name);
        if (found == columnIndex.end()) throw UserError(std::string("UCP lookup table is missing column: ") + name);
        return found->second;
    };
    const std::size_t frcCol = require("FRC_URB2D"), minCol = require("MH_URB2D_MIN"), mhCol = require("MH_URB2D"),
                       maxCol = require("MH_URB2D_MAX"), bldfrCol = require("BLDFR_URB2D"), h2wCol = require("H2W");

    std::map<int, UcpRow> table;
    std::string line;
    while (std::getline(input, line)) {
        if (trimmed(line).empty()) continue;
        const auto fields = splitCsvLine(line);
        if (fields.size() <= std::max({frcCol, minCol, mhCol, maxCol, bldfrCol, h2wCol}))
            throw UserError("UCP lookup table row has too few columns: " + line);
        const int lczClass = std::stoi(fields[0]);
        table[lczClass] = {std::stod(fields[frcCol]), std::stod(fields[minCol]), std::stod(fields[mhCol]), std::stod(fields[maxCol]),
            std::stod(fields[bldfrCol]), std::stod(fields[h2wCol])};
    }
    return table;
}

// w2w's own resources/LCZ_UCP_lookup.csv, byte-for-byte (see
// defaultUcpTable()'s doc comment in lcz.hpp) - keep in sync with
// resources/lcz_ucp_lookup.csv; the "defaultUcpTable() matches the bundled
// CSV file" core_tests.cpp case catches drift between the two.
constexpr const char* kDefaultUcpTableCsv =
    ",FRC_URB2D,MH_URB2D_MIN,MH_URB2D,MH_URB2D_MAX,BLDFR_URB2D,H2W\n"
    "1,0.95,25,50,75,0.5,2.5\n"
    "2,0.9,10,17.5,25,0.55,1.25\n"
    "3,0.85,3,6.5,10,0.55,1.25\n"
    "4,0.65,25,50,75,0.3,1\n"
    "5,0.7,10,17.5,25,0.3,0.5\n"
    "6,0.6,3,6.5,10,0.3,0.5\n"
    "7,0.85,4,5,6,0.75,1.5\n"
    "8,0.85,3,6.5,10,0.4,0.2\n"
    "9,0.3,3,6.5,10,0.15,0.15\n"
    "10,0.55,5,10,15,0.25,0.35\n"
    "11,0,0,0,0,0,0\n"
    "12,0,0,0,0,0,0\n"
    "13,0,0,0,0,0,0\n"
    "14,0,0,0,0,0,0\n"
    "15,0.95,0,0,0,0.05,0\n"
    "16,0,0,0,0,0,0\n"
    "17,0,0,0,0,0,0\n";
}  // namespace

std::map<int, UcpRow> loadUcpTable(const std::filesystem::path& csvPath) {
    std::ifstream input(csvPath);
    if (!input) throw UserError("Could not open UCP lookup table: " + csvPath.string());
    return parseUcpTable(input, csvPath.string());
}

std::map<int, UcpRow> loadUcpTableFromString(const std::string& csvContent) {
    std::istringstream input(csvContent);
    return parseUcpTable(input, "<in-memory UCP table>");
}

const std::map<int, UcpRow>& defaultUcpTable() {
    static const std::map<int, UcpRow> table = loadUcpTableFromString(kDefaultUcpTableCsv);
    return table;
}

void checkCustomUcpTableIntegrity(const std::map<int, UcpRow>& ucpTable) {
    for (int lczClass = 1; lczClass <= 10; ++lczClass) {
        const auto found = ucpTable.find(lczClass);
        if (found == ucpTable.end()) throw UserError("UCP lookup table is missing built LCZ class " + std::to_string(lczClass));
        const auto& row = found->second;
        if (!(row.mhUrb2dMin < row.mhUrb2d) || !(row.mhUrb2d < row.mhUrb2dMax))
            throw UserError("MH_URB2D_MIN (MH_URB2D_MAX) should be smaller (larger) than MH_URB2D - check the custom UCP table.");
    }
}

LczRaster checkLczIntegrity(const std::filesystem::path& lczPath, int lczBand, const NetcdfFile& wrfFile) {
    GDALAllRegister();
    std::unique_ptr<GDALDataset, void (*)(GDALDataset*)> dataset(
        GDALDataset::FromHandle(GDALOpenEx(lczPath.string().c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr)),
        [](GDALDataset* d) { if (d) GDALClose(d); });
    if (!dataset) throw UserError("Cannot find LCZ map file: " + lczPath.string());
    if (lczBand < 0 || lczBand >= dataset->GetRasterCount())
        throw UserError("Cannot read the requested LCZ band " + std::to_string(lczBand) + " from the LCZ GeoTIFF - check the -l/--lcz-band argument.");

    const int width = dataset->GetRasterXSize(), height = dataset->GetRasterYSize();
    std::vector<float> values(static_cast<std::size_t>(width) * height);
    auto* band = dataset->GetRasterBand(lczBand + 1);  // GDAL bands are 1-indexed
    if (band->RasterIO(GF_Read, 0, 0, width, height, values.data(), width, height, GDT_Float32, 0, 0) != CE_None)
        throw UserError("Could not read the LCZ GeoTIFF: " + lczPath.string());
    // warpToCrs (like the rest of this pipeline) treats NaN as nodata, but
    // GDAL doesn't convert a band's registered NoData value to NaN on
    // read - do that ourselves so a real nodata value (e.g. -1, as
    // testing/Shanghai.tif uses) is actually excluded from the warp below,
    // rather than participating in nearest-neighbor resampling as if it
    // were real LCZ class data.
    int hasNodata = 0;
    const double nodata = band->GetNoDataValue(&hasNodata);
    if (hasNodata)
        for (auto& value : values)
            if (value == static_cast<float>(nodata)) value = std::numeric_limits<float>::quiet_NaN();

    std::array<double, 6> sourceGeotransform{};
    if (dataset->GetGeoTransform(sourceGeotransform.data()) != CE_None) throw UserError("LCZ GeoTIFF has no geotransform: " + lczPath.string());
    const std::string sourceWkt = dataset->GetProjectionRef() ? dataset->GetProjectionRef() : "";

    // _replace_lcz_number (w2w.py): if any 100-series (LCZ Generator)
    // class label is present, remap the WHOLE raster through
    // {1..10 -> 1..10, 101..107 -> 11..17} - matching w2w.py's own
    // dict-based remap, including that a value outside both ranges is
    // left unchanged here (Python's pandas .map() would instead turn it
    // into NaN and then undefined-behavior-cast that to int32, which
    // isn't a deliberate design choice worth reproducing).
    const bool has100Series = std::any_of(values.begin(), values.end(), [](float v) { return v >= 101.0f && v <= 107.0f; });
    if (has100Series) {
        std::map<int, int> remap;
        for (int v = 1; v <= 10; ++v) remap[v] = v;
        for (int v = 101; v <= 107; ++v) remap[v] = v - 101 + 11;
        for (auto& value : values) {
            if (std::isnan(value)) continue;
            const auto found = remap.find(static_cast<int>(std::lround(value)));
            if (found != remap.end()) value = static_cast<float>(found->second);
        }
    }

    const std::string wgs84Wkt = canonicalWgs84Wkt();
    OGRSpatialReference sourceSrs, wgs84Srs;
    sourceSrs.importFromWkt(sourceWkt.c_str());
    wgs84Srs.importFromEPSG(4326);
    const bool alreadyWgs84 = sourceSrs.IsSame(&wgs84Srs);

    std::array<double, 6> geotransform = sourceGeotransform;
    int finalWidth = width, finalHeight = height;
    if (!alreadyWgs84) {
        auto warped = warpToCrs(values, width, height, sourceWkt, sourceGeotransform, wgs84Wkt, ResampleMethod::Nearest);
        for (auto& value : warped.values)
            if (!(value > 0.0f)) value = 0.0f;  // xr.where(lcz.data > 0, lcz.data, 0) - also clears any NaN a partial-coverage warp left behind
        finalWidth = warped.width;
        finalHeight = warped.height;
        const double dx = (warped.bounds3857.maxX - warped.bounds3857.minX) / finalWidth;
        const double dy = (warped.bounds3857.maxY - warped.bounds3857.minY) / finalHeight;
        geotransform = {warped.bounds3857.minX, dx, 0.0, warped.bounds3857.maxY, 0.0, -dy};
        values = std::move(warped.values);
    }

    // _check_lcz_wrf_extent: the LCZ raster must cover the WRF domain's
    // own XLONG_M/XLAT_M bounds in every direction (strict inequality,
    // matching w2w.py exactly).
    const double lczXmin = geotransform[0], lczYmax = geotransform[3];
    const double lczXmax = lczXmin + geotransform[1] * finalWidth, lczYmin = lczYmax + geotransform[5] * finalHeight;
    const auto wrfLon = wrfFile.readFloat("XLONG_M");
    const auto wrfLat = wrfFile.readFloat("XLAT_M");
    const auto [wrfLonMinIt, wrfLonMaxIt] = std::minmax_element(wrfLon.begin(), wrfLon.end());
    const auto [wrfLatMinIt, wrfLatMaxIt] = std::minmax_element(wrfLat.begin(), wrfLat.end());
    const double wrfXmin = *wrfLonMinIt, wrfXmax = *wrfLonMaxIt, wrfYmin = *wrfLatMinIt, wrfYmax = *wrfLatMaxIt;
    if (!(wrfXmin > lczXmin && wrfXmax < lczXmax && wrfYmin > lczYmin && wrfYmax < lczYmax))
        throw UserError(std::format("LCZ domain should be larger than the WRF domain in all directions. LCZ bounds (xmin, ymin, xmax, ymax): "
                                     "({}, {}, {}, {}); WRF bounds (xmin, ymin, xmax, ymax): ({}, {}, {}, {})",
            lczXmin, lczYmin, lczXmax, lczYmax, wrfXmin, wrfYmin, wrfXmax, wrfYmax));

    return {std::move(values), finalWidth, finalHeight, wgs84Wkt, geotransform};
}

double streetWidth(const UcpRow& row) { return row.mhUrb2d / row.h2w; }
double buildingWidth(const UcpRow& row) { return row.bldfrUrb2d / (row.frcUrb2d - row.bldfrUrb2d) * streetWidth(row); }

namespace {
// _get_lcz_arr + the replacer-array pattern every _ucp_resampler/
// _hi_resampler bin uses: a per-pixel value from `lookup` for classes
// present in it (i.e. BUILT_LCZ), 0 everywhere else (including NaN source
// pixels, and classes deliberately absent from `lookup` - e.g. LCZ 15 in
// hiResample).
std::vector<float> lookupRasterOverBuilt(const LczRaster& clean, const std::map<int, double>& lookup) {
    std::vector<float> result(clean.values.size(), 0.0f);
    for (std::size_t i = 0; i < clean.values.size(); ++i) {
        if (std::isnan(clean.values[i])) continue;
        const auto found = lookup.find(static_cast<int>(std::lround(clean.values[i])));
        if (found != lookup.end()) result[i] = static_cast<float>(found->second);
    }
    return result;
}

std::vector<float> warpAverageOntoGrid(const std::vector<float>& perPixel, const LczRaster& clean, const WrfGridInfo& grid) {
    return warpToGrid(perPixel, clean.width, clean.height, clean.wkt, clean.geotransform, grid.crs.wkt(), grid.geotransform, grid.width, grid.height,
        ResampleMethod::Average);
}

double standardNormalCdf(double z) { return 0.5 * (1.0 + std::erf(z / std::numbers::sqrt2)); }

// The 15 five-meter HI_URB2D bin percentages (0-75m) for one LCZ class's
// truncated-normal building-height distribution - see hiResample's doc
// comment on why this is computed analytically rather than by replaying
// w2w.py's own Monte Carlo sampling. All-zero for a degenerate
// distribution (MH_URB2D_MAX <= MH_URB2D_MIN - LCZ 15's default row is
// exactly this: 0/0/0), matching _compute_hi_distribution's explicit
// `if not i == 15` skip.
std::array<double, 15> hiDistributionForClass(const UcpRow& row) {
    std::array<double, 15> bins{};
    const double lo = row.mhUrb2dMin, hi = row.mhUrb2dMax, mean = row.mhUrb2d;
    if (!(hi > lo)) return bins;
    const double sd = (hi - lo) / 4.0;
    const double zLo = (lo - mean) / sd, zHi = (hi - mean) / sd;
    const double denom = standardNormalCdf(zHi) - standardNormalCdf(zLo);
    for (int b = 0; b < 15; ++b) {
        const double e0 = b * 5.0, e1 = (b + 1) * 5.0;
        const double c0 = std::clamp(e0, lo, hi), c1 = std::clamp(e1, lo, hi);
        if (c1 <= c0) continue;
        const double za = (c0 - mean) / sd, zb = (c1 - mean) / sd;
        bins[static_cast<std::size_t>(b)] = 100.0 * (standardNormalCdf(zb) - standardNormalCdf(za)) / denom;
    }
    return bins;
}

// The per-month GREENFRAC average over the ORIGINAL (pre-any-processing)
// file's own urban pixels, plus that file's NUM_LAND_CAT/ISURBAN -
// _adjust_greenfrac_landusef's `dst_data_orig`-derived values.
struct OrigUrbanInfo {
    std::array<double, 12> greenfracPerMonth{};
    int origNumLandCat{};
    int urbanCat{};
};
OrigUrbanInfo computeOrigUrbanInfo(const NetcdfFile& orig) {
    OrigUrbanInfo info;
    info.origNumLandCat = static_cast<int>(globalAttr(orig, "NUM_LAND_CAT"));
    info.urbanCat = static_cast<int>(globalAttr(orig, "ISURBAN"));

    std::set<int> urbanCatSet{info.urbanCat};
    if (info.origNumLandCat == 61)
        for (int c = 51; c <= 60; ++c) urbanCatSet.insert(c);
    else if (info.origNumLandCat == 41)
        for (int c = 31; c <= 40; ++c) urbanCatSet.insert(c);

    const auto lu = orig.readFloat("LU_INDEX");
    const auto luShape = orig.shape("LU_INDEX");
    const std::size_t npix = luShape[1] * luShape[2];
    std::vector<bool> wrfUrb(npix);
    for (std::size_t p = 0; p < npix; ++p) wrfUrb[p] = urbanCatSet.contains(static_cast<int>(std::lround(lu[p])));

    const auto greenf = orig.readFloat("GREENFRAC");
    for (int mm = 0; mm < 12; ++mm) {
        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t p = 0; p < npix; ++p)
            if (wrfUrb[p]) { sum += greenf[static_cast<std::size_t>(mm) * npix + p]; ++count; }
        info.greenfracPerMonth[static_cast<std::size_t>(mm)] = count > 0 ? sum / static_cast<double>(count) : 0.0;
    }
    return info;
}
}  // namespace

std::vector<float> ucpResample(const LczRaster& clean, const WrfGridInfo& grid, const std::vector<int>& builtLcz, const std::map<int, UcpRow>& ucpTable,
    UcpKey key, std::optional<double> frcThreshold) {
    std::map<int, double> lookup;
    for (int cls : builtLcz) {
        const auto& row = ucpTable.at(cls);
        double value{};
        switch (key) {
            case UcpKey::FrcUrb2d: value = row.frcUrb2d; break;
            case UcpKey::MhUrb2d: value = row.mhUrb2d; break;
            case UcpKey::StdhUrb2d: value = (row.mhUrb2dMax - row.mhUrb2dMin) / 4.0; break;
            case UcpKey::LbUrb2d:
            case UcpKey::LfUrb2d:
            case UcpKey::LpUrb2d: {
                const double sw = streetWidth(row), bw = buildingWidth(row);
                const double lambdaP = bw / (bw + sw);
                const double lambdaF = 2.0 * row.mhUrb2d / (bw + sw);
                value = key == UcpKey::LpUrb2d ? lambdaP : key == UcpKey::LfUrb2d ? lambdaF : lambdaP + lambdaF;
                break;
            }
        }
        lookup[cls] = value;
    }

    auto warped = warpAverageOntoGrid(lookupRasterOverBuilt(clean, lookup), clean, grid);
    for (auto& v : warped) {
        if (frcThreshold && !(v > *frcThreshold)) v = 0.0f;
        if (std::isnan(v)) v = 0.0f;
    }
    return warped;
}

std::vector<float> hgtResample(const LczRaster& clean, const WrfGridInfo& grid, const std::vector<int>& builtLcz, const std::map<int, UcpRow>& ucpTable) {
    std::map<int, double> lookupNom, lookupDenom;
    for (int cls : builtLcz) {
        const auto& row = ucpTable.at(cls);
        const double bw = buildingWidth(row);
        lookupNom[cls] = bw * bw * row.mhUrb2d;
        lookupDenom[cls] = bw * bw;
    }
    const auto nom = warpAverageOntoGrid(lookupRasterOverBuilt(clean, lookupNom), clean, grid);
    const auto denom = warpAverageOntoGrid(lookupRasterOverBuilt(clean, lookupDenom), clean, grid);

    std::vector<float> result(nom.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        const float v = nom[i] / denom[i];
        result[i] = std::isnan(v) ? 0.0f : v;
    }
    return result;
}

std::vector<float> lczResample(const LczRaster& clean, const WrfGridInfo& grid, const std::vector<int>& builtLcz, int addLczInt) {
    const std::set<int> built(builtLcz.begin(), builtLcz.end());
    std::vector<float> masked(clean.values.size());
    for (std::size_t i = 0; i < clean.values.size(); ++i) {
        const float v = clean.values[i];
        masked[i] = (std::isnan(v) || !built.contains(static_cast<int>(std::lround(v)))) ? std::numeric_limits<float>::quiet_NaN() : v;
    }
    auto warped = warpToGrid(masked, clean.width, clean.height, clean.wkt, clean.geotransform, grid.crs.wkt(), grid.geotransform, grid.width, grid.height,
        ResampleMethod::Mode);

    const bool builtHas15 = built.contains(15);
    for (auto& v : warped) {
        if (std::isnan(v)) continue;
        int cls = static_cast<int>(std::lround(v));
        if (cls == 15 && builtHas15) cls = 11;
        v = static_cast<float>(cls + addLczInt);
    }
    return warped;
}

HiResampleResult hiResample(const LczRaster& clean, const WrfGridInfo& grid, const std::vector<int>& builtLcz, const std::map<int, UcpRow>& ucpTable) {
    std::map<int, std::array<double, 15>> distributions;
    for (int cls : builtLcz)
        if (cls != 15) distributions[cls] = hiDistributionForClass(ucpTable.at(cls));

    const std::size_t npix = static_cast<std::size_t>(grid.width) * grid.height;
    std::vector<float> hiArr(15 * npix, 0.0f);
    for (int b = 0; b < 15; ++b) {
        std::map<int, double> lookup;
        for (const auto& [cls, bins] : distributions) lookup[cls] = bins[static_cast<std::size_t>(b)];
        const auto warped = warpAverageOntoGrid(lookupRasterOverBuilt(clean, lookup), clean, grid);
        for (std::size_t p = 0; p < npix; ++p) {
            float v = warped[p];
            if (std::isnan(v)) v = 0.0f;
            if (v < 5.0f) v = 0.0f;  // HI_THRES_MIN
            hiArr[static_cast<std::size_t>(b) * npix + p] = v;
        }
    }

    for (std::size_t p = 0; p < npix; ++p) {
        double sum = 0.0;
        for (int b = 0; b < 15; ++b) sum += hiArr[static_cast<std::size_t>(b) * npix + p];
        if (sum > 0.0)
            for (int b = 0; b < 15; ++b)
                hiArr[static_cast<std::size_t>(b) * npix + p] = static_cast<float>(hiArr[static_cast<std::size_t>(b) * npix + p] / sum * 100.0);
    }

    int nbuiMax = 0;
    for (std::size_t p = 0; p < npix; ++p) {
        int count = 0;
        for (int b = 0; b < 15; ++b)
            if (hiArr[static_cast<std::size_t>(b) * npix + p] != 0.0f) ++count;
        nbuiMax = std::max(nbuiMax, count);
    }
    return {std::move(hiArr), nbuiMax};
}

int createLczParamsFile(const LczParamsInputs& inputs, const std::filesystem::path& outPath) {
    const auto grid = wrfGridInfo(NetcdfFile::open(inputs.noUrbanPath, NetcdfFile::Mode::ReadOnly));
    const std::size_t npix = static_cast<std::size_t>(grid.width) * grid.height;

    auto frcUrb2d = ucpResample(inputs.clean, grid, inputs.builtLcz, inputs.ucpTable, UcpKey::FrcUrb2d, inputs.frcThreshold);
    std::vector<bool> frcMask(npix);
    for (std::size_t p = 0; p < npix; ++p) frcMask[p] = frcUrb2d[p] != 0.0f;

    const auto lczResampled = lczResample(inputs.clean, grid, inputs.builtLcz, inputs.wrfVersion.addLczInt);
    const auto mhUrb2d = ucpResample(inputs.clean, grid, inputs.builtLcz, inputs.ucpTable, UcpKey::MhUrb2d);
    const auto stdhUrb2d = ucpResample(inputs.clean, grid, inputs.builtLcz, inputs.ucpTable, UcpKey::StdhUrb2d);
    const auto lbUrb2d = ucpResample(inputs.clean, grid, inputs.builtLcz, inputs.ucpTable, UcpKey::LbUrb2d);
    const auto lfUrb2d = ucpResample(inputs.clean, grid, inputs.builtLcz, inputs.ucpTable, UcpKey::LfUrb2d);
    const auto lpUrb2d = ucpResample(inputs.clean, grid, inputs.builtLcz, inputs.ucpTable, UcpKey::LpUrb2d);
    const auto hgtUrb2d = hgtResample(inputs.clean, grid, inputs.builtLcz, inputs.ucpTable);
    const auto hi = hiResample(inputs.clean, grid, inputs.builtLcz, inputs.ucpTable);

    const auto origInfo = computeOrigUrbanInfo(NetcdfFile::open(inputs.origPath, NetcdfFile::Mode::ReadOnly));

    NetcdfFile::copyFile(inputs.noUrbanPath, outPath);

    const int numLandCat = inputs.wrfVersion.numLandCat;
    std::vector<float> newLuIndex, newLandusef;
    {
        auto dst = NetcdfFile::open(outPath, NetcdfFile::Mode::ReadWrite);

        newLuIndex = dst.readFloat("LU_INDEX");
        for (std::size_t p = 0; p < npix; ++p)
            if (frcMask[p]) newLuIndex[p] = lczResampled[p];
        dst.writeFloat("LU_INDEX", newLuIndex);

        auto greenf = dst.readFloat("GREENFRAC");
        for (int mm = 0; mm < 12; ++mm)
            for (std::size_t p = 0; p < npix; ++p)
                if (frcMask[p]) greenf[static_cast<std::size_t>(mm) * npix + p] = static_cast<float>(origInfo.greenfracPerMonth[static_cast<std::size_t>(mm)]);
        dst.writeFloat("GREENFRAC", greenf);

        // LANDUSEF's full replacement content, built entirely up front
        // (both the copy-existing-categories pass and the arange(31,
        // numLandCat) single-category pass, since `newLuIndex` - the
        // POST-LCZ-write LU_INDEX - is already known) so the structural
        // rebuild below can fold it in alongside FRC_URB2D/URB_PARAM in
        // one pass, rather than resizing first and reopening to finish it.
        const auto preResizeLanduseF = dst.readFloat("LANDUSEF");
        newLandusef.assign(static_cast<std::size_t>(numLandCat) * npix, 0.0f);
        const std::size_t copyCats = std::min(static_cast<std::size_t>(origInfo.origNumLandCat), preResizeLanduseF.size() / npix);
        for (std::size_t cat = 0; cat < copyCats; ++cat)
            for (std::size_t p = 0; p < npix; ++p)
                if (!frcMask[p]) newLandusef[cat * npix + p] = preResizeLanduseF[cat * npix + p];
        // np.arange(31, numLandCat, 1) - literally 31, not 30+1, regardless
        // of ADD_LCZ_INT: a real w2w.py quirk (see lcz.hpp's
        // createLczParamsFile doc), kept as-is rather than "fixed" to
        // start at ADD_LCZ_INT+1.
        for (int luValue = 31; luValue < numLandCat; ++luValue)
            for (std::size_t p = 0; p < npix; ++p)
                if (static_cast<int>(std::lround(newLuIndex[p])) == luValue) newLandusef[static_cast<std::size_t>(luValue - 1) * npix + p] = 1.0f;
    }

    std::vector<float> urbParam(132 * npix, 0.0f);
    const auto place = [&](int zeroIndexedSlot, const std::vector<float>& values) {
        for (std::size_t p = 0; p < npix; ++p) urbParam[static_cast<std::size_t>(zeroIndexedSlot) * npix + p] = frcMask[p] ? values[p] : 0.0f;
    };
    place(90, lpUrb2d);
    place(91, mhUrb2d);
    place(92, stdhUrb2d);
    place(93, hgtUrb2d);
    place(94, lbUrb2d);
    for (int i = 0; i < 4; ++i) place(95 + i, lfUrb2d);
    for (int b = 0; b < 15; ++b) {
        std::vector<float> bin(npix);
        for (std::size_t p = 0; p < npix; ++p) bin[p] = hi.hiUrb2d[static_cast<std::size_t>(b) * npix + p];
        place(117 + b, bin);
    }

    // One rebuild pass: resizes land_cat (LANDUSEF) AND (re)defines
    // FRC_URB2D/URB_PARAM from scratch - some real geo_em files already
    // carry both from geogrid's own default (non-LCZ) urban
    // parameterization (confirmed against w2w's own sample_data/
    // geo_em.d04.nc: it already has FRC_URB2D and an 18-slot URB_PARAM),
    // which plain defineVariable can't redefine in place.
    NetcdfFile::rebuildStructure(
        outPath, "land_cat", static_cast<std::size_t>(numLandCat),
        [&](const std::string& variableName) -> std::optional<std::vector<float>> {
            return variableName == "LANDUSEF" ? std::optional(newLandusef) : std::nullopt;
        },
        {{"num_urb_params", 132}},
        {
            {"FRC_URB2D", NC_FLOAT, {"Time", "south_north", "west_east"}, frcUrb2d},
            {"URB_PARAM", NC_FLOAT, {"Time", "num_urb_params", "south_north", "west_east"}, urbParam},
        });

    {
        auto dst = NetcdfFile::open(outPath, NetcdfFile::Mode::ReadWrite);
        const auto setTextAttr = [&](const std::string& var, const std::string& name, const std::string& text) {
            NetcdfFile::Attribute a;
            a.name = name;
            a.type = NC_CHAR;
            a.text = text;
            dst.putAttribute(var, a);
        };
        const auto setIntAttr = [&](const std::string& var, const std::string& name, int value) {
            NetcdfFile::Attribute a;
            a.name = name;
            a.type = NC_INT;
            a.numbers = {static_cast<double>(value)};
            dst.putAttribute(var, a);
        };
        setIntAttr("FRC_URB2D", "FieldType", 104);
        setTextAttr("FRC_URB2D", "MemoryOrder", "XY");
        setTextAttr("FRC_URB2D", "units", "-");
        setTextAttr("FRC_URB2D", "description", "ufrac");
        setTextAttr("FRC_URB2D", "stagger", "M");
        setIntAttr("FRC_URB2D", "sr_x", 1);
        setIntAttr("FRC_URB2D", "sr_y", 1);

        NetcdfFile::Attribute titleAttr;
        if (dst.hasAttribute("", "TITLE")) {
            titleAttr = dst.getAttribute("", "TITLE");
            titleAttr.text += ", perturbed by W2W";
            dst.putAttribute("", titleAttr);
        }
        const auto setGlobalText = [&](const std::string& name, const std::string& text) {
            NetcdfFile::Attribute a;
            a.name = name;
            a.type = NC_CHAR;
            a.text = text;
            dst.putAttribute("", a);
        };
        const auto setGlobalInt = [&](const std::string& name, int value) {
            NetcdfFile::Attribute a;
            a.name = name;
            a.type = NC_INT;
            a.numbers = {static_cast<double>(value)};
            dst.putAttribute("", a);
        };
        setGlobalInt("NUM_LAND_CAT", numLandCat);
        setGlobalInt("FLAG_URB_PARAM", 1);
        setGlobalInt("NBUI_MAX", hi.nbuiMax);
        setGlobalText("DESCRIPTION",
            "W2W.py tool used to create geo_em*.nc file:\n Demuzere, M., Argueso, D., Zonato, A., & Kittner, J. (2021). \n"
            "W2W: A Python package that injects WUDAPT's Local Climate Zone \n"
            "information in WRF [Computer software]. \n"
            "https://github.com/matthiasdemuzere/w2w");
    }

    return hi.nbuiMax;
}

void createLczExtentFile(const std::filesystem::path& paramsPath, const std::filesystem::path& origPath, const std::filesystem::path& outPath) {
    auto params = NetcdfFile::open(paramsPath, NetcdfFile::Mode::ReadOnly);
    const int paramsNumLandCat = static_cast<int>(globalAttr(params, "NUM_LAND_CAT"));
    auto luIndex = params.readFloat("LU_INDEX");
    const auto luShape = params.shape("LU_INDEX");
    const std::size_t npix = luShape[1] * luShape[2];
    std::vector<bool> frcMask(npix);
    {
        const auto frcUrb2d = params.readFloat("FRC_URB2D");
        for (std::size_t p = 0; p < npix; ++p) frcMask[p] = frcUrb2d[p] != 0.0f;
    }

    auto orig = NetcdfFile::open(origPath, NetcdfFile::Mode::ReadOnly);
    const int origNumLandCat = static_cast<int>(globalAttr(orig, "NUM_LAND_CAT"));
    const int urbanCat = static_cast<int>(globalAttr(orig, "ISURBAN"));

    // Matches w2w.py's own if/elif exactly: neither branch (an
    // unrecognized NUM_LAND_CAT on the params file) leaves LU_INDEX
    // untouched, rather than guessing a threshold.
    if (paramsNumLandCat == 61 || paramsNumLandCat == 41) {
        const int lczThreshold = paramsNumLandCat == 61 ? 51 : 31;
        for (auto& v : luIndex)
            if (static_cast<int>(std::lround(v)) >= lczThreshold) v = static_cast<float>(urbanCat);
    }

    const auto landusefFull = params.readFloat("LANDUSEF");
    std::vector<float> landusef(static_cast<std::size_t>(origNumLandCat) * npix, 0.0f);
    for (std::size_t cat = 0; cat < static_cast<std::size_t>(origNumLandCat); ++cat)
        for (std::size_t p = 0; p < npix; ++p) landusef[cat * npix + p] = landusefFull[cat * npix + p];
    // Explicitly clear any stale LCZ-range fraction beyond the LU_INDEX
    // rollback above (add_wrf_version-only guard).
    const int lufZeroFrom = origNumLandCat == 61 ? 50 : origNumLandCat == 41 ? 30 : origNumLandCat;
    for (std::size_t cat = static_cast<std::size_t>(lufZeroFrom); cat < static_cast<std::size_t>(origNumLandCat); ++cat)
        for (std::size_t p = 0; p < npix; ++p) landusef[cat * npix + p] = 0.0f;
    for (std::size_t p = 0; p < npix; ++p)
        if (frcMask[p]) landusef[static_cast<std::size_t>(urbanCat - 1) * npix + p] = 1.0f;

    NetcdfFile::copyFile(paramsPath, outPath);
    NetcdfFile::resizeDimension(outPath, "land_cat", static_cast<std::size_t>(origNumLandCat),
        [&](const std::string& variableName) -> std::optional<std::vector<float>> { return variableName == "LANDUSEF" ? std::optional(landusef) : std::nullopt; });

    auto dst = NetcdfFile::open(outPath, NetcdfFile::Mode::ReadWrite);
    dst.writeFloat("LU_INDEX", luIndex);
    NetcdfFile::Attribute flagAttr;
    flagAttr.name = "FLAG_URB_PARAM";
    flagAttr.type = NC_INT;
    flagAttr.numbers = {0.0};
    dst.putAttribute("", flagAttr);
    NetcdfFile::Attribute numCatAttr;
    numCatAttr.name = "NUM_LAND_CAT";
    numCatAttr.type = NC_INT;
    numCatAttr.numbers = {static_cast<double>(origNumLandCat)};
    dst.putAttribute("", numCatAttr);
    // FRC_URB2D/URB_PARAM are copied along with everything else by
    // copyFile/resizeDimension and then never referenced again - w2w.py
    // explicitly drops them (`dst_extent.drop_vars(['FRC_URB2D',
    // 'URB_PARAM'])`) to reduce file size, which this port doesn't
    // replicate (NetcdfFile has no variable-removal primitive); leaving
    // them present but unused is a size-only difference, not a
    // correctness one.
}

const std::vector<WrfVersionOption>& wrfVersionOptions() {
    static const std::vector<WrfVersionOption> options = {
        {"v4.3", {30, 41}},
        {"v4.3.1", {30, 41}},
        {"v4.3.2", {30, 41}},
        {"v4.3.3", {30, 41}},
        {"v4.4", {30, 41}},
        {"v4.4.1", {30, 41}},
        {"v4.4.2", {50, 61}},
        {"v4.5", {50, 61}},
        {"v4.5.1", {50, 61}},
        {"v4.5.2", {50, 61}},
    };
    return options;
}

std::vector<std::string> expandLandCatParents(const std::filesystem::path& dstFile, const WrfVersionInfo& wrfVersion) {
    const std::string full = dstFile.string();
    const std::string domainDigits = full.size() >= 5 ? full.substr(full.size() - 5, 2) : std::string();
    if (domainDigits.size() != 2 || !std::isdigit(static_cast<unsigned char>(domainDigits[0])) || !std::isdigit(static_cast<unsigned char>(domainDigits[1])))
        throw UserError("Not a geo_em.dNN.nc file path (expected a two-digit domain number before .nc): " + full);
    const int domainNr = std::stoi(domainDigits);
    const std::string prefix = full.substr(0, full.size() - 5);

    std::vector<std::string> messages;
    for (int i = 1; i < domainNr; ++i) {
        const std::string ifile = std::format("{}{:02d}.nc", prefix, i);
        if (!std::filesystem::exists(ifile)) {
            messages.push_back(std::format(
                "WARNING: Parent domain {} not found. Please make sure the parent domain files are present. "
                "Without this information, you will not be able to produce the boundary conditions with real.exe.",
                ifile));
            continue;
        }

        auto src = NetcdfFile::open(ifile, NetcdfFile::Mode::ReadOnly);
        if (!src.hasAttribute("", "NUM_LAND_CAT") || !src.hasVariable("LANDUSEF")) {
            messages.push_back("Cannot read NUM_LAND_CAT and LANDUSEF dimensions for " + ifile);
            continue;
        }

        const int origNumLandCat = static_cast<int>(globalAttr(src, "NUM_LAND_CAT"));
        if (origNumLandCat == wrfVersion.numLandCat) {
            messages.push_back(std::format("Parent domain {} already contains {} LC classes", ifile, wrfVersion.numLandCat));
            continue;
        }

        const auto shape = src.shape("LANDUSEF");  // Time, land_cat, south_north, west_east
        const std::size_t ny = shape[2];
        const std::size_t nx = shape[3];
        const std::size_t npix = ny * nx;
        const auto landusef = src.readFloat("LANDUSEF");
        std::vector<float> landusefNew(static_cast<std::size_t>(wrfVersion.numLandCat) * npix, 0.0f);
        const std::size_t copyCats = std::min(static_cast<std::size_t>(origNumLandCat), shape[1]);
        for (std::size_t cat = 0; cat < copyCats; ++cat)
            for (std::size_t p = 0; p < npix; ++p) landusefNew[cat * npix + p] = landusef[cat * npix + p];

        const std::string description = origNumLandCat < 24
            ? std::format("Noah-modified {}-category IGBP-MODIS landuse", wrfVersion.numLandCat)
            : std::format("modified {}-category USGS landuse", wrfVersion.numLandCat);

        const std::string ofile = ifile.substr(0, ifile.size() - 3) + std::format("_{}.nc", wrfVersion.numLandCat);
        NetcdfFile::copyFile(ifile, ofile);
        NetcdfFile::resizeDimension(ofile, "land_cat", static_cast<std::size_t>(wrfVersion.numLandCat),
            [&](const std::string& variableName) -> std::optional<std::vector<float>> {
                return variableName == "LANDUSEF" ? std::optional(landusefNew) : std::nullopt;
            });

        auto dst = NetcdfFile::open(ofile, NetcdfFile::Mode::ReadWrite);
        NetcdfFile::Attribute descAttr;
        descAttr.name = "description";
        descAttr.type = NC_CHAR;
        descAttr.text = description;
        dst.putAttribute("LANDUSEF", descAttr);
        NetcdfFile::Attribute numCatAttr;
        numCatAttr.name = "NUM_LAND_CAT";
        numCatAttr.type = NC_INT;
        numCatAttr.numbers = {static_cast<double>(wrfVersion.numLandCat)};
        dst.putAttribute("", numCatAttr);

        messages.push_back(std::format("Parent domain {} expanded to {} LC classes -> {}", ifile, wrfVersion.numLandCat, ofile));
    }
    return messages;
}

namespace {

// pandas Series.std() default (ddof=1, sample standard deviation).
double sampleStd(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double sumSq = 0.0;
    for (double v : values) sumSq += (v - mean) * (v - mean);
    return std::sqrt(sumSq / static_cast<double>(values.size() - 1));
}

// _check_range: fails (returns false) if any value in `darr` falls outside
// [lo, hi] inclusive.
bool valuesWithinRange(const std::vector<float>& darr, double lo, double hi) {
    for (float v : darr)
        if (v < lo || v > hi) return false;
    return true;
}

}  // namespace

std::vector<CheckResult> checksAndCleaning(const ChecksAndCleaningInputs& inputs, const std::map<int, UcpRow>& ucpTable) {
    auto dstDataOrig = NetcdfFile::open(inputs.dstFile, NetcdfFile::Mode::ReadOnly);
    const int origNumLandCat = static_cast<int>(globalAttr(dstDataOrig, "NUM_LAND_CAT"));
    const int urbanCat = static_cast<int>(globalAttr(dstDataOrig, "ISURBAN"));

    std::vector<int> lczUrban;
    std::vector<int> urbanCatList;
    if (origNumLandCat == 61) {
        lczUrban = {51, 52, 53, 54, 55, 56, 57, 58, 59, 60};
        urbanCatList = lczUrban;
        urbanCatList.push_back(urbanCat);
    } else if (origNumLandCat == 41) {
        lczUrban = {31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
        urbanCatList = lczUrban;
        urbanCatList.push_back(urbanCat);
    } else if (origNumLandCat == 21 || origNumLandCat == 20) {
        lczUrban = {};
        urbanCatList = {urbanCat};
    } else {
        throw UserError("Number of land categories " + std::to_string(origNumLandCat) + " in original file not supported. Only 21 (20), 41 or 61 are supported.");
    }

    std::vector<CheckResult> results;
    const auto containsAny = [](const std::vector<float>& values, const std::vector<int>& set) {
        for (float v : values)
            if (std::find(set.begin(), set.end(), static_cast<int>(std::lround(v))) != set.end()) return true;
        return false;
    };

    // Check 1
    {
        auto da = NetcdfFile::open(inputs.dstNuFile, NetcdfFile::Mode::ReadOnly);
        const bool stillUrban = containsAny(da.readFloat("LU_INDEX"), urbanCatList);
        results.push_back({"Check 1: Urban class removed from " + inputs.dstNuFile.filename().string() + "?",
            stillUrban ? CheckStatus::Warning : CheckStatus::Ok, stillUrban ? "Urban land use still present" : "OK"});
    }

    // Check 2
    {
        auto da = NetcdfFile::open(inputs.dstLczExtentFile, NetcdfFile::Mode::ReadOnly);
        const bool present = containsAny(da.readFloat("LU_INDEX"), {urbanCat});
        results.push_back({"Check 2: LCZ Urban extent present in " + inputs.dstLczExtentFile.filename().string() + "?",
            present ? CheckStatus::Ok : CheckStatus::Warning, present ? "OK" : "LCZ-based urban extent missing"});
    }

    // Check 3
    bool paramsUrbanStillPresent = false;
    {
        auto da = NetcdfFile::open(inputs.dstLczParamsFile, NetcdfFile::Mode::ReadOnly);
        const auto luIndex = da.readFloat("LU_INDEX");
        paramsUrbanStillPresent = containsAny(luIndex, {urbanCat});
        if (paramsUrbanStillPresent) {
            results.push_back({"Check 3: Urban LCZ classes exists in " + inputs.dstLczParamsFile.filename().string() + "?", CheckStatus::Warning,
                std::format("Urban extent still defined via LU_INDEX = {}?", urbanCat)});
        } else {
            std::set<int> uniqueValues;
            for (float v : luIndex) uniqueValues.insert(static_cast<int>(std::lround(v)));
            std::vector<int> presentLczs;
            for (int cls : lczUrban)
                if (uniqueValues.contains(cls)) presentLczs.push_back(cls);
            std::string list;
            for (std::size_t i = 0; i < presentLczs.size(); ++i) list += (i ? ", " : "") + std::to_string(presentLczs[i]);
            results.push_back({"Check 3: Urban LCZ classes exists in " + inputs.dstLczParamsFile.filename().string() + "?", CheckStatus::Ok,
                std::format("OK: LCZ Classes ({}) present", list)});
        }
    }

    // Check 4
    bool frcUrb2dPresent = false;
    {
        auto da = NetcdfFile::open(inputs.dstLczParamsFile, NetcdfFile::Mode::ReadOnly);
        frcUrb2dPresent = da.hasVariable("FRC_URB2D");
        if (frcUrb2dPresent) {
            const auto frc = da.readFloat("FRC_URB2D");
            const auto [minIt, maxIt] = std::minmax_element(frc.begin(), frc.end());
            results.push_back({"Check 4: FRC_URB2D present in " + inputs.dstLczParamsFile.filename().string() + "?", CheckStatus::Ok,
                std::format("OK: FRC_URB2D values range between {:.2f} and {:.2f}", *minIt, *maxIt)});
        } else {
            results.push_back({"Check 4: FRC_URB2D present in " + inputs.dstLczParamsFile.filename().string() + "?", CheckStatus::Warning,
                "FRC_URB2D not present"});
        }
    }

    // Check 5
    bool urbParamPresent = false;
    {
        auto da = NetcdfFile::open(inputs.dstLczParamsFile, NetcdfFile::Mode::ReadOnly);
        urbParamPresent = da.hasVariable("URB_PARAM");
        results.push_back({"Check 5: URB_PARAMS matrix present in file " + inputs.dstLczParamsFile.filename().string() + "?",
            urbParamPresent ? CheckStatus::Ok : CheckStatus::Warning, urbParamPresent ? "OK" : "URB_PARAM matrix not present"});
    }

    if (urbParamPresent) {
        auto da = NetcdfFile::open(inputs.dstLczParamsFile, NetcdfFile::Mode::ReadOnly);
        const auto urbParam = da.readFloat("URB_PARAM");
        const auto shape = da.shape("URB_PARAM");  // Time, num_urb_params, south_north, west_east
        const std::size_t npix = shape[2] * shape[3];
        const auto slice = [&](int zeroIndexedSlot) {
            return std::vector<float>(urbParam.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(zeroIndexedSlot) * npix),
                urbParam.begin() + static_cast<std::ptrdiff_t>((static_cast<std::size_t>(zeroIndexedSlot) + 1) * npix));
        };

        std::vector<double> mhUrb2dCol, mhUrb2dMaxCol;
        for (const auto& [cls, row] : ucpTable) {
            mhUrb2dCol.push_back(row.mhUrb2d);
            mhUrb2dMaxCol.push_back(row.mhUrb2dMax);
        }
        const double mhMax = *std::max_element(mhUrb2dCol.begin(), mhUrb2dCol.end());
        const double mhStd = sampleStd(mhUrb2dCol);
        const double mhMaxMax = *std::max_element(mhUrb2dMaxCol.begin(), mhUrb2dMaxCol.end());

        // Check 6 - URB PAR indices per NUDAPT_44 documentation (1-based in
        // the doc, zero-based slot here): LP=90, MH=91, STDH=92, HGT=93,
        // LB=94, LF=95. STDH_URB2D and HGT_URB2D's ranges both reuse
        // MH_URB2D-derived bounds rather than their own column - ported
        // as-is from w2w.py, not "fixed".
        struct UcpCheck {
            std::string key;
            int slot;
            double lo, hi;
        };
        const std::vector<UcpCheck> ucpChecks = {
            {"LP_URB2D", 90, 0.0, 1.0},
            {"MH_URB2D", 91, 0.0, mhMax + mhStd},
            {"STDH_URB2D", 92, 0.0, mhMaxMax + mhStd},
            {"HGT_URB2D", 93, 0.0, mhMax + mhStd},
            {"LB_URB2D", 94, 0.0, 5.0},
            {"LF_URB2D", 95, 0.0, 5.0},
        };
        bool check6Ok = true;
        std::string check6Message;
        for (const auto& c : ucpChecks) {
            const bool ok = valuesWithinRange(slice(c.slot), c.lo, c.hi);
            if (!ok) check6Ok = false;
            check6Message += (check6Message.empty() ? "" : "; ") + c.key + (ok ? ": OK" : ": exceeds expected value range");
        }
        results.push_back({"Check 6: Do URB_PARAM variable values follow expected range in " + inputs.dstLczParamsFile.filename().string() + "?",
            check6Ok ? CheckStatus::Ok : CheckStatus::Warning, check6Message});

        // Check 7 - HI_URB2D (slots 117..131) must sum to 100% at every
        // pixel that has any nonzero bin.
        double maxAbsDeviation = 0.0;
        bool anyUrbanPixel = false;
        for (std::size_t p = 0; p < npix; ++p) {
            double sum = 0.0;
            for (int b = 0; b < 15; ++b) sum += urbParam[static_cast<std::size_t>(117 + b) * npix + p];
            if (sum == 0.0) continue;
            anyUrbanPixel = true;
            maxAbsDeviation = std::max(maxAbsDeviation, std::abs(100.0 - sum));
        }
        const bool check7Ok = !anyUrbanPixel || maxAbsDeviation <= 0.1;
        results.push_back({"Check 7: Does HI_URB2D sum to 100% for urban pixels in " + inputs.dstLczParamsFile.filename().string() + "?",
            check7Ok ? CheckStatus::Ok : CheckStatus::Warning, check7Ok ? "OK" : "Not all pixels have sum HI_URB2D == 100%"});
    }

    if (frcUrb2dPresent) {
        auto da = NetcdfFile::open(inputs.dstLczParamsFile, NetcdfFile::Mode::ReadOnly);
        const auto frcUrb2d = da.readFloat("FRC_URB2D");
        const auto luIndex = da.readFloat("LU_INDEX");
        // Hardcoded >= 31 regardless of orig_num_land_cat - a real w2w.py
        // quirk (checks_and_cleaning does NOT gate this one on the
        // LCZ_URBAN list, unlike checks 3 and 9), ported as-is.
        long diff = 0;
        for (std::size_t p = 0; p < frcUrb2d.size(); ++p) {
            const int frcRes = frcUrb2d[p] != 0.0f ? 1 : 0;
            const int luRes = static_cast<int>(std::lround(luIndex[p])) >= 31 ? 1 : 0;
            diff += frcRes - luRes;
        }
        const bool check8Ok = diff == 0;
        results.push_back({"Check 8: Do FRC_URB and LCZs (from LU_INDEX) cover same extent in " + inputs.dstLczParamsFile.filename().string() + "?",
            check8Ok ? CheckStatus::Ok : CheckStatus::Warning, check8Ok ? "OK" : "FRC_URB and LCZs in LU_INDEX do not cover same extent"});
    }

    // Check 9
    {
        auto daE = NetcdfFile::open(inputs.dstLczExtentFile, NetcdfFile::Mode::ReadOnly);
        auto daP = NetcdfFile::open(inputs.dstLczParamsFile, NetcdfFile::Mode::ReadOnly);
        const auto luE = daE.readFloat("LU_INDEX");
        const auto luP = daP.readFloat("LU_INDEX");
        long sumE = 0, sumP = 0, diff = 0;
        for (std::size_t p = 0; p < luE.size(); ++p) {
            const int eRes = static_cast<int>(std::lround(luE[p])) == urbanCat ? 1 : 0;
            const bool inLczUrban = std::find(lczUrban.begin(), lczUrban.end(), static_cast<int>(std::lround(luP[p]))) != lczUrban.end();
            const int pRes = inLczUrban ? 1 : 0;
            sumE += eRes;
            sumP += pRes;
            diff += pRes - eRes;
        }
        const bool check9Ok = diff == 0;
        results.push_back({"Check 9: Extent and # urban pixels same for *_extent.nc and *_params.nc output file?",
            check9Ok ? CheckStatus::Ok : CheckStatus::Warning,
            check9Ok ? std::format("OK, urban extent the same ({})", sumP)
                     : std::format("Different # urban pixels (or extent) according to LU_INDEX: extent: {}, params: {}", sumE, sumP)});
    }

    if (std::filesystem::exists(inputs.srcFileClean)) std::filesystem::remove(inputs.srcFileClean);

    return results;
}

}  // namespace wrftools
