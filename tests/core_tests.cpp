#include "wrftools/domain.hpp"
#include "wrftools/error.hpp"
#include "wrftools/wrf_series.hpp"
#include "wrftools/wrf_file.hpp"
#include "wrftools/wps_namelist.hpp"
#include "wrftools/colormaps.hpp"
#include "wrftools/units.hpp"
#include "wrftools/raster_layer.hpp"
#include "wrftools/warp.hpp"
#include "wrftools/layer_renderer.hpp"
#include "wrftools/colorbar.hpp"
#include "wrftools/wps_binary_source.hpp"
#include "wrftools/wrf_source.hpp"
#include "wrftools/netcdf_file.hpp"
#include "wrftools/derived_variable.hpp"
#include "wrftools/lcz.hpp"
#include "wrftools/reproject.hpp"
#include "wrftools/stats.hpp"
#include "fast_exit.hpp"

#include <netcdf.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <limits>
#include <memory>
#include <set>
#include <sstream>

#include <gdal_priv.h>
#include <ogr_spatialref.h>

using namespace wrftools;

// A custom main (rather than linking Catch2's auto-provided one) purely to
// call fastExit below - see its comment. Otherwise identical to what
// Catch2WithMain itself generates.
int main(int argc, char* argv[]) {
    const int result = Catch::Session().run(argc, argv);
    wrftools_tests::fastExit(result);  // see fast_exit.hpp
}

TEST_CASE("WRF series names are parsed and ordered") {
    const auto parsed = parseWrfFilename("wrfout_d03_2025-03-14_00_30_00");
    REQUIRE(parsed);
    CHECK(parsed->kind == "wrfout");
    CHECK(parsed->domain == "03");
    const auto grouped = groupWrfPaths({"wrfout_d03_2025-03-14_01_00_00", "wrfout_d03_2025-03-14_00_30_00", "geo_em.d01.nc"});
    REQUIRE(grouped.groups.size() == 1);
    CHECK(grouped.groups[0][0].filename() == "wrfout_d03_2025-03-14_00_30_00");
    CHECK(grouped.singles.size() == 1);
}

TEST_CASE("WRF naming accepts every supported separator and rejects invalid names") {
    const auto colon = parseWrfFilename("wrfout_d03_2025-03-14_00:30:00");
    REQUIRE(colon);
    const auto met = parseWrfFilename("met_em.d01.2025-03-14_00:00:00.nc");
    REQUIRE(met);
    CHECK(met->kind == "met_em");
    CHECK_FALSE(parseWrfFilename("geo_em.d01.nc"));
    CHECK_FALSE(parseWrfFilename("wrfinput_d01"));
    CHECK_FALSE(parseWrfFilename("wrfout_d03_2025-02-30_00_00_00"));
    CHECK_FALSE(parseWrfFilename("wrfout_d03_2025-03-14_25_00_00"));
}

TEST_CASE("WRF grouping keeps domains separate and lone files single") {
    const auto grouped = groupWrfPaths({"wrfout_d01_2020-01-01_00_00_00", "wrfout_d02_2020-01-01_00_00_00", "wrfout_d01_2020-01-01_00_30_00", "geo_em.d01.nc"});
    REQUIRE(grouped.groups.size() == 1);
    CHECK(grouped.groups.front().size() == 2);
    REQUIRE(grouped.singles.size() == 2);
    CHECK(std::find(grouped.singles.begin(), grouped.singles.end(), "wrfout_d02_2020-01-01_00_00_00") != grouped.singles.end());
}

TEST_CASE("WRF series opens lazily and reads a selected timestamp") {
    WrfFileSeries series({"tests/fixtures/wrfout_d01_2020-01-01_00_00_00.nc", "tests/fixtures/wrfout_d01_2020-01-01_00_30_00.nc", "tests/fixtures/wrfout_d01_2020-01-01_01_00_00.nc"});
    CHECK(series.openedFileCount() == 1);
    REQUIRE(series.times().size() == 3);
    CHECK(series.times()[1] == "2020-01-01 00:30");
    CHECK(series.read("T2", 1).size() == 64);
    CHECK(series.openedFileCount() == 2);
    CHECK_THROWS_AS(series.read("T2", 3), UserError);
}

TEST_CASE("grouping edge cases: lone recognized file and input-order independence") {
    const auto lone = groupWrfPaths({"tests/fixtures/wrfout_d01_2020-01-01_00_00_00.nc"});
    CHECK(lone.groups.empty());
    REQUIRE(lone.singles.size() == 1);

    const std::vector<std::filesystem::path> ordered{
        "tests/fixtures/wrfout_d01_2020-01-01_00_00_00.nc", "tests/fixtures/wrfout_d01_2020-01-01_00_30_00.nc",
        "tests/fixtures/wrfout_d01_2020-01-01_01_00_00.nc"};
    auto reversed = ordered;
    std::reverse(reversed.begin(), reversed.end());
    const auto grouped = groupWrfPaths(reversed);
    REQUIRE(grouped.groups.size() == 1);
    CHECK(grouped.groups.front() == ordered);
}

TEST_CASE("WRF series read matches reading the underlying file directly") {
    const std::vector<std::filesystem::path> paths{
        "tests/fixtures/wrfout_d01_2020-01-01_00_00_00.nc", "tests/fixtures/wrfout_d01_2020-01-01_00_30_00.nc",
        "tests/fixtures/wrfout_d01_2020-01-01_01_00_00.nc"};
    WrfFileSeries series(paths);
    WrfFile direct("tests/fixtures/wrfout_d01_2020-01-01_00_30_00.nc");
    CHECK(series.read("T2", 1) == direct.read("T2", 0, 0));
}

TEST_CASE("WRF series name and path describe the group") {
    const std::vector<std::filesystem::path> paths{
        "tests/fixtures/wrfout_d01_2020-01-01_00_00_00.nc", "tests/fixtures/wrfout_d01_2020-01-01_00_30_00.nc",
        "tests/fixtures/wrfout_d01_2020-01-01_01_00_00.nc"};
    WrfFileSeries series(paths);
    CHECK(series.name() == "wrfout_d01 (3 files, 2020-01-01 00:00 - 2020-01-01 01:00)");
    CHECK(series.path() == paths.front());
}

TEST_CASE("WRF series rereading the same file does not reopen it") {
    WrfFileSeries series({"tests/fixtures/wrfout_d01_2020-01-01_00_00_00.nc", "tests/fixtures/wrfout_d01_2020-01-01_00_30_00.nc", "tests/fixtures/wrfout_d01_2020-01-01_01_00_00.nc"});
    CHECK(series.openedFileCount() == 1);
    CHECK_NOTHROW(series.read("T2", 0));
    CHECK(series.openedFileCount() == 1);
    CHECK(series.isFileOpen(1) == false);
}

TEST_CASE("WRF series rejects a mismatched grid at construction when it must open eagerly") {
    // wrfout_with_1d_var.nc's name doesn't match the series pattern, so
    // pairing it with a real series member forces the eager-open fallback,
    // which validates every file's grid up front.
    CHECK_THROWS_AS(WrfFileSeries({"tests/fixtures/wrfout_d01_2020-01-01_00_00_00.nc", "tests/fixtures/wrfout_with_1d_var.nc"}), UserError);
}

TEST_CASE("WRF series grid mismatch several files in is only caught when read") {
    // wrfout_d01_2020-01-01_02_00_00.nc parses as a real series member (so
    // construction takes the fast lazy path and does not open it) but
    // actually shares wrfout_with_1d_var.nc's different grid - the
    // mismatch only surfaces once that file is actually read.
    WrfFileSeries series({
        "tests/fixtures/wrfout_d01_2020-01-01_00_00_00.nc", "tests/fixtures/wrfout_d01_2020-01-01_00_30_00.nc",
        "tests/fixtures/wrfout_d01_2020-01-01_01_00_00.nc", "tests/fixtures/wrfout_d01_2020-01-01_02_00_00.nc"});
    CHECK(series.openedFileCount() == 1);
    CHECK_NOTHROW(series.read("T2", 0));
    CHECK_NOTHROW(series.read("T2", 2));
    CHECK_THROWS_AS(series.read("T2", 3), UserError);
}

TEST_CASE("WRF series with a multi-timestep file falls back to eager open") {
    // wrfout_multitime.nc's own filename doesn't match the series pattern
    // at all, so pairing it with a real series member exercises the "can't
    // trust the filename for timestep count" fallback path.
    WrfFileSeries series({"tests/fixtures/wrfout_multitime.nc", "tests/fixtures/wrfout_d01_2020-01-01_00_00_00.nc"});
    CHECK(series.openedFileCount() == 2);
    CHECK(series.isFileOpen(0));
    CHECK(series.isFileOpen(1));
}

TEST_CASE("domain projects accept siblings and reject invalid parent ids") {
    DomainProject project({
        Domain{.id = 1, .parentId = 1, .ratio = 1, .columns = 100, .rows = 100},
        Domain{.id = 2, .parentId = 1, .ratio = 3, .columns = 30, .rows = 30},
        Domain{.id = 3, .parentId = 1, .ratio = 3, .columns = 30, .rows = 30},
    });
    CHECK(project.domains().size() == 3);
    CHECK_THROWS_AS(DomainProject({Domain{.id = 1, .parentId = 1, .ratio = 1, .columns = 10, .rows = 10}, Domain{.id = 2, .parentId = 2, .ratio = 3, .columns = 3, .rows = 3}}), UserError);
}

TEST_CASE("domain containment and subtree removal are validated") {
    Domain parent{.id = 1, .parentId = 1, .columns = 100, .rows = 100, .bounds = Bounds{0, 0, 100, 100}};
    Domain child{.id = 2, .parentId = 1, .ratio = 3, .columns = 10, .rows = 10, .bounds = Bounds{10, 10, 20, 20}};
    Domain sibling{.id = 3, .parentId = 1, .ratio = 3, .columns = 10, .rows = 10, .bounds = Bounds{30, 30, 40, 40}};
    Domain grandchild{.id = 4, .parentId = 2, .ratio = 3, .columns = 5, .rows = 5, .bounds = Bounds{12, 12, 15, 15}};
    DomainProject project({parent, child, sibling, grandchild});
    project.removeSubtree(2);
    REQUIRE(project.domains().size() == 2);
    CHECK(project.domains()[1].id == 2);
    CHECK(project.domains()[1].parentId == 1);
    child.bounds = Bounds{90, 90, 110, 110};
    CHECK_THROWS_AS(DomainProject({parent, child}), UserError);
}

TEST_CASE("GDAL-backed reader discovers fixture variables") {
    WrfFile file("tests/fixtures/geo_em_small.nc");
    REQUIRE_FALSE(file.variables().empty());
    CHECK(std::any_of(file.variables().begin(), file.variables().end(), [](const WrfVariable& value) { return value.name == "LU_INDEX"; }));
    CHECK(file.geographicBounds().north > file.geographicBounds().south);
    CHECK(file.geographicBounds().east > file.geographicBounds().west);
    CHECK_FALSE(file.projectionWkt().empty());
}

// Pinned against the Python reference (wrftools.wrfreader.WRFFile) run
// against the same fixtures: `uv run python -c "from wrftools.wrfreader
// import WRFFile; f = WRFFile(path); print(f.geotransform)"`. Both fixtures
// are Mercator (MAP_PROJ=3) - exactly the projection the old
// srs.SetMercator(truelat1, standLon, 1.0, ...) construction got wrong
// (truelat1 belongs at lat_ts, not as a scale-1 origin latitude).
TEST_CASE("geotransform matches the Python reference for Mercator fixtures") {
    for (const char* path : {"tests/fixtures/geo_em_small.nc", "tests/fixtures/wrfout_multitime.nc"}) {
        WrfFile file(path);
        const auto gt = file.geotransform();
        CHECK(gt[0] == Catch::Approx(-3118.452108972693).epsilon(1e-6));
        CHECK(gt[1] == Catch::Approx(250.05003010343663).epsilon(1e-6));
        CHECK(gt[2] == 0.0);
        CHECK(gt[3] == Catch::Approx(2353541.4751173565).epsilon(1e-6));
        CHECK(gt[4] == 0.0);
        CHECK(gt[5] == Catch::Approx(-249.9877820990514).epsilon(1e-6));
        CHECK(file.projectionWkt().find("Mercator") != std::string::npos);
    }
}

TEST_CASE("GDAL-backed reader reads real multi-time rasters") {
    WrfFile file("tests/fixtures/wrfout_multitime.nc");
    REQUIRE(std::any_of(file.variables().begin(), file.variables().end(), [](const WrfVariable& value) { return value.name == "T2"; }));
    const auto first = file.read("T2", 1);
    const auto second = file.read("T2", 2);
    REQUIRE(first.size() == 64);
    REQUIRE(second.size() == 64);
    const auto firstMean = std::accumulate(first.begin(), first.end(), 0.0) / static_cast<double>(first.size());
    const auto secondMean = std::accumulate(second.begin(), second.end(), 0.0) / static_cast<double>(second.size());
    CHECK(secondMean > firstMean);
    CHECK_THROWS_AS(file.read("T2", 99), UserError);
    CHECK_THROWS_AS(file.read("NOT_A_REAL_VARIABLE"), UserError);
}

TEST_CASE("GDAL-backed reader reports dimensions and destaggers wind fields") {
    WrfFile file("tests/fixtures/wrfout_multitime.nc");
    CHECK(file.size() == std::array<int, 2>{8, 8});
    const auto wind = std::find_if(file.variables().begin(), file.variables().end(), [](const WrfVariable& value) { return value.name == "U"; });
    REQUIRE(wind != file.variables().end());
    CHECK(wind->timeCount == 3);
    CHECK(wind->levelCount == 3);
    const auto values = file.read("U", 0, 0);
    CHECK(values.size() == 64);
    CHECK_THROWS_AS(file.read("U", 0, 3), UserError);
}

// Ported from tests/test_wrfreader.py's remaining cases not already
// covered above.
TEST_CASE("geo_em fixture exposes exactly its two static 2D variables") {
    WrfFile file("tests/fixtures/geo_em_small.nc");
    std::set<std::string> names;
    for (const auto& variable : file.variables()) names.insert(variable.name);
    CHECK(names == std::set<std::string>{"HGT_M", "LU_INDEX"});
    for (const auto& variable : file.variables()) {
        CHECK_FALSE(variable.extraDimension.has_value());
        CHECK(variable.timeCount == 1);
        CHECK(variable.levelCount == 1);
    }
}

// Regression coverage for a real bug: GDAL exposes 1D vertical-only
// variables (ZS/DZS, MemoryOrder "Z  ") as tiny (N,1) "rasters" that
// corrupted mass-grid-size inference and silently dropped every real 2D
// variable from a genuine wrfout file - see wrf_file.cpp's MemoryOrder
// filter.
TEST_CASE("1D vertical-only variables are excluded and don't break 2D ones") {
    WrfFile file("tests/fixtures/wrfout_with_1d_var.nc");
    const auto& variables = file.variables();
    CHECK_FALSE(std::any_of(variables.begin(), variables.end(), [](const WrfVariable& v) { return v.name == "ZS"; }));
    CHECK_FALSE(std::any_of(variables.begin(), variables.end(), [](const WrfVariable& v) { return v.name == "DZS"; }));
    CHECK(std::any_of(variables.begin(), variables.end(), [](const WrfVariable& v) { return v.name == "T2"; }));
    CHECK(std::any_of(variables.begin(), variables.end(), [](const WrfVariable& v) { return v.name == "HGT"; }));
    CHECK(file.size() == std::array<int, 2>{8, 8});
}

TEST_CASE("WrfFile rejects an unsupported MAP_PROJ") {
    CHECK_THROWS_AS(WrfFile("tests/fixtures/wrf_unknown_projection.nc"), UnsupportedError);
}

TEST_CASE("WrfFile rejects a rotated-pole lat/lon projection") {
    CHECK_THROWS_AS(WrfFile("tests/fixtures/wrf_rotated_pole.nc"), UnsupportedError);
}

TEST_CASE("WrfFile rejects a GDAL-openable file with no MAP_PROJ attribute") {
    CHECK_THROWS_AS(WrfFile("tests/fixtures/not_a_wrf_file.tif"), UserError);
}

TEST_CASE("WrfFile reads a NetCDF4/HDF5-backed file identically to its classic-format original") {
    // wrfout_multitime_nc4.nc is `nccopy -k nc4` of wrfout_multitime.nc -
    // same data, HDF5-backed storage. Some real production WRF output is
    // NetCDF4/HDF5-backed, and on a machine whose GDAL netCDF driver isn't
    // built with HDF5 support, a bare (non-driver-forced) open would hand
    // such a file to GDAL's generic HDF5 driver instead, which exposes
    // variables so differently (attributes on band metadata instead of
    // dataset metadata, HDF5:"path"://VAR subdataset names) that every
    // variable used to get filtered out entirely. WrfFile always forces
    // the netCDF driver via the NETCDF: prefix specifically to avoid this -
    // see wrf_file.cpp's constructor comment.
    WrfFile classic("tests/fixtures/wrfout_multitime.nc");
    WrfFile nc4("tests/fixtures/wrfout_multitime_nc4.nc");
    CHECK(nc4.variables().size() == classic.variables().size());
    CHECK(nc4.read("T2", 1, 0) == classic.read("T2", 1, 0));
}

TEST_CASE("WrfFile tags LU_INDEX with an LCZ-aware category scheme only for an actual *_LCZ_params.nc file") {
    const auto luIndexScheme = [](const std::string& path) {
        WrfFile file(path);
        for (const auto& variable : file.variables())
            if (variable.name == "LU_INDEX") return variable.categoryScheme.value_or("<none>");
        FAIL("no LU_INDEX variable in " << path);
        return std::string();
    };

    // The real *_LCZ_params.nc file - the only one whose LU_INDEX actually
    // carries WUDAPT LCZ class numbers - gets the combined scheme name.
    CHECK(luIndexScheme("tests/fixtures/lcz/geo_em.d04_LCZ_params.nc") == "MODIFIED_IGBP_MODIS_NOAH+LCZ41");

    // *_LCZ_extent.nc copies createLczParamsFile's own DESCRIPTION marker
    // forward AND (for this specific fixture pair) coincidentally still
    // has NUM_LAND_CAT=41 too (the original geo_em.d04.nc already had 41
    // categories from geogrid's own default urban physics, before w2w
    // ever touched it - see PORT_W2W.MD Stage 3) - FLAG_URB_PARAM=0 is
    // what correctly excludes it despite both other signals matching.
    CHECK(luIndexScheme("tests/fixtures/lcz/geo_em.d04_LCZ_extent.nc") == "MODIFIED_IGBP_MODIS_NOAH");

    // A plain geo_em file w2w never touched (no DESCRIPTION marker at
    // all) is unaffected - the pre-existing, unchanged behavior.
    CHECK(luIndexScheme("tests/fixtures/lcz/geo_em.d04.nc") == "MODIFIED_IGBP_MODIS_NOAH");
}

TEST_CASE("level index selects distinct data") {
    WrfFile file("tests/fixtures/wrfout_multitime.nc");
    std::set<long> roundedMeans;
    for (int level = 0; level < 3; ++level) {
        const auto values = file.read("U", 0, level);
        const auto mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
        roundedMeans.insert(std::lround(mean * 1000));
    }
    CHECK(roundedMeans.size() == 3);
}

TEST_CASE("destaggering averages adjacent staggered cells") {
    // Reads U's raw (undestaggered) band directly via a separate GDAL
    // handle and checks WrfFile::read's destaggered output against the
    // textbook definition: the average of each pair of adjacent staggered
    // cells along the staggered axis.
    GDALAllRegister();
    const std::string target = "NETCDF:\"tests/fixtures/wrfout_multitime.nc\":U";
    std::unique_ptr<GDALDataset, decltype(&GDALClose)> raw(
        static_cast<GDALDataset*>(GDALOpenEx(target.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr)), GDALClose);
    REQUIRE(raw);
    const int width = raw->GetRasterXSize(), height = raw->GetRasterYSize();
    std::vector<float> rawValues(static_cast<std::size_t>(width) * height);
    REQUIRE(raw->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, rawValues.data(), width, height, GDT_Float32, 0, 0) == CE_None);

    WrfFile file("tests/fixtures/wrfout_multitime.nc");
    const auto destaggered = file.read("U", 0, 0);
    REQUIRE(destaggered.size() == static_cast<std::size_t>(height) * (width - 1));
    for (int y = 0; y < height; ++y) for (int x = 0; x < width - 1; ++x) {
        const float expected = (rawValues[static_cast<std::size_t>(y) * width + x] + rawValues[static_cast<std::size_t>(y) * width + x + 1]) / 2.0f;
        CHECK(destaggered[static_cast<std::size_t>(y) * (width - 1) + x] == Catch::Approx(expected).epsilon(1e-5));
    }
}

TEST_CASE("WPS sibling fixture imports and exports") {
    const auto project = readWpsNamelist("tests/fixtures/namelist_siblings.wps");
    REQUIRE(project.domains.domains().size() == 4);
    CHECK(project.domains.domains()[2].parentId == 2);
    CHECK(project.domains.domains()[3].parentId == 2);
    const auto output = std::filesystem::temp_directory_path() / "wrftools-cpp-namelist.wps";
    writeWpsNamelist(project, output);
    CHECK(readWpsNamelist(output).domains.domains().size() == 4);
    std::filesystem::remove(output);
}

// Pinned against gis4wrf.core.Project.bboxes for the same fixture:
// `uv run python -c "from gis4wrf.core.readers.namelist import
// read_namelist; from gis4wrf.core.transforms.wps_namelist_to_project
// import convert_wps_nml_to_project; from gis4wrf.core.project import
// Project; nml = read_namelist(path, schema_name='wps'); project =
// convert_wps_nml_to_project(nml, Project.create()); print([bbox for bbox
// in project.bboxes])"` - domains 3/4 are Lambert siblings sharing domain
// 2 as their parent, exactly the tree shape fillDomains's single
// forward pass has to get right.
TEST_CASE("fillDomains bboxes match the Python reference for a sibling tree") {
    auto project = readWpsNamelist("tests/fixtures/namelist_siblings.wps");
    project.domains.fillDomains();
    const auto& domains = project.domains.domains();
    const std::array<Bounds, 4> expected{{
        {-281249.99999999994, -284375.0000000014, 281250.00000000006, 284374.9999999986},
        {-156249.99999999994, -159375.0000000014, 156250.00000000006, 159374.9999999986},
        {15000.000000000058, 8124.999999998603, 65000.00000000006, 59374.9999999986},
        {-61249.99999999994, -64375.0000000014, -11249.999999999942, -13125.000000001397},
    }};
    for (std::size_t i = 0; i < domains.size(); ++i) {
        REQUIRE(domains[i].bounds.has_value());
        CHECK(domains[i].bounds->minX == Catch::Approx(expected[i].minX).margin(1e-3));
        CHECK(domains[i].bounds->minY == Catch::Approx(expected[i].minY).margin(1e-3));
        CHECK(domains[i].bounds->maxX == Catch::Approx(expected[i].maxX).margin(1e-3));
        CHECK(domains[i].bounds->maxY == Catch::Approx(expected[i].maxY).margin(1e-3));
    }
}

// Ported from tests/test_crs_datum.py's
// test_root_domain_corner_lonlat_matches_sphere_datum, which exists to
// pin the WRF-sphere datum choice (radius 6370000 m, not WGS84) - see that
// file's module docstring for why a bbox CORNER, not the center, is the
// point that actually exercises the datum (a projection's own origin
// always round-trips to itself regardless of datum). These exact values
// are the Python suite's own pinned constants, not re-derived here.
TEST_CASE("root domain bbox corners match the Python reference (WRF sphere datum)") {
    auto project = readWpsNamelist("tests/fixtures/namelist_siblings.wps");
    project.domains.fillDomains();
    const auto& root = project.domains.domains().front();
    REQUIRE(root.bounds.has_value());
    const auto projection = project.domains.projection();
    const auto bottomLeft = projection.toLonLat({root.bounds->minX, root.bounds->minY});
    const auto topRight = projection.toLonLat({root.bounds->maxX, root.bounds->maxY});
    CHECK(bottomLeft.lon == Catch::Approx(3.112809).margin(1e-5));
    CHECK(bottomLeft.lat == Catch::Approx(43.905622).margin(1e-5));
    CHECK(topRight.lon == Catch::Approx(10.476400).margin(1e-5));
    CHECK(topRight.lat == Catch::Approx(49.014080).margin(1e-5));
}

TEST_CASE("WPS export preserves nesting and root projection fields") {
    const auto original = readWpsNamelist("tests/fixtures/namelist_siblings.wps");
    const auto output = std::filesystem::temp_directory_path() / "wrftools-cpp-roundtrip.wps";
    writeWpsNamelist(original, output);
    const auto reread = readWpsNamelist(output);
    CHECK(reread.domains.domains().front().mapProj == original.domains.domains().front().mapProj);
    CHECK(std::abs(reread.domains.domains().front().centerLon - original.domains.domains().front().centerLon) < 1e-12);
    CHECK(reread.domains.domains()[1].ratio == original.domains.domains()[1].ratio);
    CHECK(reread.domains.domains()[3].paddingBottom == original.domains.domains()[3].paddingBottom);
    std::filesystem::remove(output);
}

namespace {
void checkBboxRoundTrip(const std::string& fixture) {
    auto original = readWpsNamelist(fixture);
    original.domains.fillDomains();
    const auto output = std::filesystem::temp_directory_path() / "wrftools-cpp-bbox-roundtrip.wps";
    writeWpsNamelist(original, output);
    auto reimported = readWpsNamelist(output);
    reimported.domains.fillDomains();
    std::filesystem::remove(output);

    const auto& originalDomains = original.domains.domains();
    const auto& reimportedDomains = reimported.domains.domains();
    REQUIRE(reimportedDomains.size() == originalDomains.size());
    for (std::size_t i = 0; i < originalDomains.size(); ++i) {
        CHECK(reimportedDomains[i].parentId == originalDomains[i].parentId);
        REQUIRE(originalDomains[i].bounds.has_value());
        REQUIRE(reimportedDomains[i].bounds.has_value());
        CHECK(reimportedDomains[i].bounds->minX == Catch::Approx(originalDomains[i].bounds->minX).margin(1.0));
        CHECK(reimportedDomains[i].bounds->minY == Catch::Approx(originalDomains[i].bounds->minY).margin(1.0));
        CHECK(reimportedDomains[i].bounds->maxX == Catch::Approx(originalDomains[i].bounds->maxX).margin(1.0));
        CHECK(reimportedDomains[i].bounds->maxY == Catch::Approx(originalDomains[i].bounds->maxY).margin(1.0));
    }
}
}  // namespace

TEST_CASE("sibling fixture export round-trips its domain bboxes") { checkBboxRoundTrip("tests/fixtures/namelist_siblings.wps"); }
TEST_CASE("linear-chain fixture export round-trips its domain bboxes") { checkBboxRoundTrip("tests/fixtures/namelist_hongkong.wps"); }

TEST_CASE("WPS parser gives user errors for malformed domain cardinality") {
    const auto path = std::filesystem::temp_directory_path() / "wrftools-cpp-invalid.wps";
    std::ofstream out(path);
    out << "&share\n max_dom = 2,\n/\n&geogrid\n parent_id = 1,\n parent_grid_ratio = 1,\n i_parent_start = 1,\n j_parent_start = 1,\n e_we = 10,\n e_sn = 10,\n map_proj = 'mercator',\n dx = 1000,\n dy = 1000,\n ref_lon = 0,\n ref_lat = 0,\n/\n";
    out.close();
    CHECK_THROWS_AS(readWpsNamelist(path), UserError);
    std::filesystem::remove(path);
}

TEST_CASE("WPS parser handles array values wrapped across lines") {
    // tests/fixtures/namelist_wrapped.wps deliberately wraps parent_id and
    // j_parent_start mid-array onto a continuation line with no '=' on it -
    // valid Fortran namelist syntax the old line-oriented-only parser
    // silently truncated (it would see "parent_id = 1," as a complete
    // one-element array and drop the wrapped "1," entirely).
    const auto project = readWpsNamelist("tests/fixtures/namelist_wrapped.wps");
    const auto& domains = project.domains.domains();
    REQUIRE(domains.size() == 2);
    CHECK(domains[0].parentId == 1);
    CHECK(domains[1].parentId == 1);
    CHECK(domains[1].paddingLeft == 9);   // i_parent_start = 10 -> 0-based 9
    CHECK(domains[1].paddingBottom == 9); // j_parent_start wrapped across two lines
    CHECK(domains[1].columns == 60);      // e_we = 61 -> domain_size 60
}

TEST_CASE("WPS linear-chain fixture remains valid") {
    const auto project = readWpsNamelist("tests/fixtures/namelist_hongkong.wps");
    REQUIRE(project.domains.domains().size() == 3);
    CHECK(project.domains.domains()[1].parentId == 1);
    CHECK(project.domains.domains()[2].parentId == 2);
}

TEST_CASE("unit conversions match WRF display choices") {
    const auto temperature = conversionsFor(" K ");
    REQUIRE(temperature.size() == 3);
    CHECK(convert(273.15, findUnit("K", "degC")) == Catch::Approx(0.0));
    std::vector<float> wind{1.0f, 2.0f};
    convertInPlace(wind, findUnit("m/s", "kmh"));
    CHECK(wind == std::vector<float>{3.6f, 7.2f});
    CHECK_THROWS_AS(findUnit("K", "kn"), std::out_of_range);
}

// Ported from tests/test_units.py's remaining cases not already covered
// above.
TEST_CASE("unit conversions: unrecognized/blank/category units offer only native") {
    CHECK(conversionsFor("some-made-up-unit").size() == 1);
    CHECK(conversionsFor("").size() == 1);
    CHECK(conversionsFor("category").size() == 1);
    CHECK(conversionsFor("some-made-up-unit").front().key == "native");
}

TEST_CASE("unit conversions: Kelvin to Fahrenheit boiling point") {
    CHECK(convert(373.15, findUnit("K", "degF")) == Catch::Approx(212.0));
}

TEST_CASE("unit conversions: m/s to knots matches the pinned conversion factor") {
    CHECK(convert(1.0, findUnit("m s-1", "kn")) == Catch::Approx(1.9438444924406046));
}

TEST_CASE("unit conversions: CF-style m/s spelling variants all normalize the same") {
    for (const std::string& spelling : {"m s-1", "m/s", "ms-1", "M S-1", "  m   s-1  "}) {
        const auto options = conversionsFor(spelling);
        std::vector<std::string> keys;
        for (const auto& option : options) keys.push_back(option.key);
        std::sort(keys.begin(), keys.end());
        CHECK(keys == std::vector<std::string>{"kmh", "kn", "mph", "native"});
    }
}

TEST_CASE("unit conversions: round-trip native to target and back is lossless") {
    const auto unit = findUnit("Pa", "hpa");
    const double nativeValue = 101325.0;
    const double converted = convert(nativeValue, unit);
    const double back = (converted - unit.offset) / unit.scale;
    CHECK(back == Catch::Approx(nativeValue));
}

// Ported from tests/test_colorbar.py's _format_tick cases - pure string
// formatting, no QApplication needed.
TEST_CASE("colorbar tick formatting matches the Python reference") {
    CHECK(formatColorbarTick(291.123456) == "291");
    CHECK(formatColorbarTick(0.000123456) == "0.000123");
    CHECK(formatColorbarTick(3.14159, "fixed", 2) == "3.14");
    CHECK(formatColorbarTick(3.14159, "fixed", 0) == "3");
    CHECK(formatColorbarTick(12345.0, "scientific", 2) == "1.23e+04");
}

TEST_CASE("colormaps preserve end points and transparent no-data") {
    const auto& lut = colormap("viridis");
    CHECK(lut.front() == Rgb{68, 1, 84});
    CHECK(lut.back() == Rgb{253, 231, 37});
    const std::vector<float> values{0.0f, 0.5f, 1.0f, std::numeric_limits<float>::quiet_NaN()};
    const auto image = applyColormap(values, 0, 1, lut);
    CHECK(image.front() == Rgba{68, 1, 84, 255});
    CHECK(image[2] == Rgba{253, 231, 37, 255});
    CHECK(image.back()[3] == 0);
}

TEST_CASE("categorical colormap indexes classes directly") {
    const std::vector<float> values{1.0f, 2.0f, -1.0f, std::numeric_limits<float>::quiet_NaN()};
    const auto legend = categoricalLut("USGS", 1, 2);
    const auto pixels = applyCategoricalColormap(values, legend.lut);
    CHECK(pixels[0][3] == 255);
    CHECK(pixels[1] != pixels[0]);
    CHECK(pixels[2][3] == 0);
    CHECK(pixels[3][3] == 0);
}

// Ported from tests/test_colormaps.py's remaining cases not already
// covered above.
TEST_CASE("colormaps: unknown name is rejected") {
    CHECK_THROWS_AS(colormap("not-a-real-colormap"), std::out_of_range);
}

TEST_CASE("colormaps: degenerate range is fully transparent") {
    const auto& lut = colormap("viridis");
    const std::vector<float> values{1.0f, 2.0f};
    const auto image = applyColormap(values, 5.0f, 5.0f, lut);
    CHECK(image[0][3] == 0);
    CHECK(image[1][3] == 0);
}

TEST_CASE("colormaps: out-of-range values clip to the LUT ends") {
    const auto& lut = colormap("viridis");
    const std::vector<float> values{-100.0f, 100.0f};
    const auto image = applyColormap(values, 0.0f, 10.0f, lut);
    CHECK(Rgb{image[0][0], image[0][1], image[0][2]} == lut.front());
    CHECK(Rgb{image[1][0], image[1][1], image[1][2]} == lut.back());
}

TEST_CASE("colormaps: jet starts blue and ends red") {
    const auto& lut = colormap("jet");
    CHECK(lut.front()[2] > lut.front()[0]);  // blue > red at the low end
    CHECK(lut.back()[0] > lut.back()[2]);    // red > blue at the high end
}

TEST_CASE("categorical colormap: unknown-scheme fallback is deterministic across calls") {
    const auto first = categoricalLut("SOME_UNKNOWN_SCHEME", 1, 5);
    const auto second = categoricalLut("SOME_UNKNOWN_SCHEME", 1, 5);
    for (int value = 1; value <= 5; ++value) {
        CHECK(first.labels.at(value) == "Category " + std::to_string(value));
        CHECK(first.lut[static_cast<std::size_t>(value)] == second.lut[static_cast<std::size_t>(value)]);
    }
}

TEST_CASE("categorical colormap: out-of-LUT-range index is transparent") {
    const auto legend = categoricalLut("MODIFIED_IGBP_MODIS_NOAH", 1, 20);
    const std::vector<float> values{1.0f, 2.0f, std::numeric_limits<float>::quiet_NaN(), 999.0f};
    const auto pixels = applyCategoricalColormap(values, legend.lut);
    CHECK(Rgb{pixels[0][0], pixels[0][1], pixels[0][2]} == legend.lut[1]);
    CHECK(Rgb{pixels[1][0], pixels[1][1], pixels[1][2]} == legend.lut[2]);
    CHECK(pixels[2][3] == 0);  // NaN
    CHECK(pixels[3][3] == 0);  // out of LUT range
}

// Pinned against gis4wrf.core.readers.categories.LANDUSE: `uv run python -c
// "from gis4wrf.core.readers.categories import LANDUSE;
// print(LANDUSE['USGS'][1])"` -> ('Urban and Built-Up Land', '#FF0000').
TEST_CASE("categorical LANDUSE table matches the Python reference for known values") {
    const auto usgs = categoricalLut("USGS", 1, 3);
    CHECK(usgs.labels.at(1) == "Urban and Built-Up Land");
    CHECK(usgs.lut[1] == Rgb{0xFF, 0x00, 0x00});
    CHECK(usgs.labels.at(2) == "Dryland Cropland and Pasture");

    const auto modis = categoricalLut("MODIFIED_IGBP_MODIS_NOAH", 1, 1);
    CHECK(modis.labels.at(1) == "Evergreen Needleleaf Forest");
    CHECK(modis.lut[1] == Rgb{0x00, 0x80, 0x00});

    // An unknown scheme (e.g. a soil-type field, which has no table) falls
    // back to a generated label/color rather than failing.
    const auto soil = categoricalLut("", 5, 5);
    CHECK(soil.labels.at(5) == "Category 5");
}

TEST_CASE("categoricalLut overlays WUDAPT's LCZ palette at the right offset for a <scheme>+LCZ41/61 name") {
    const auto lcz41 = categoricalLut("MODIFIED_IGBP_MODIS_NOAH+LCZ41", 1, 40);
    // Base-scheme categories below the LCZ offset are untouched.
    CHECK(lcz41.labels.at(1) == "Evergreen Needleleaf Forest");
    CHECK(lcz41.labels.at(13) == "Urban and Built-Up");
    // LCZ 1 (built) lands at 30 + 1 = 31, with WUDAPT's own label/color -
    // NOT USGS/MODIFIED_IGBP_MODIS_NOAH's own (different!) meaning for 31.
    CHECK(lcz41.labels.at(31) == "LCZ 1: Compact high-rise");
    CHECK(lcz41.lut[31] == Rgb{0x91, 0x06, 0x13});
    CHECK(lcz41.labels.at(40) == "LCZ 10: Heavy industry");

    const auto lcz61 = categoricalLut("USGS+LCZ61", 1, 60);
    CHECK(lcz61.labels.at(51) == "LCZ 1: Compact high-rise");
    CHECK(lcz61.labels.at(60) == "LCZ 10: Heavy industry");
    // USGS genuinely defines its own 31-33 ("Low/High Intensity
    // Residential"/"Industrial or Commercial") - the +LCZ61 overlay only
    // applies at 51-67, so those stay USGS's own meaning, unshadowed.
    CHECK(lcz61.labels.at(31) == "Low Intensity Residential");

    // A scheme name this table doesn't recognize at all (LCZ-suffixed or
    // not) still falls back cleanly, matching the plain-scheme case.
    const auto unknown = categoricalLut("SOMETHING+LCZ41", 5, 5);
    CHECK(unknown.labels.at(5) == "Category 5");
}

// Pinned against `wrftools.rasterlayer._warp_to_web_mercator` run on the
// same fixture/variable/time: `uv run python -c "from wrftools.rasterlayer
// import _warp_to_web_mercator; from wrftools.wrfreader import WRFFile; f =
// WRFFile(path); print(_warp_to_web_mercator(f.read('T2', 1, 0), f.crs.wkt,
// f.geotransform))"`.
TEST_CASE("warped raster bounds match the Python reference") {
    WrfFile file("tests/fixtures/wrfout_multitime.nc");
    const auto values = file.read("T2", 1, 0);
    const auto size = file.size();
    const auto warped = warpToWebMercator(values, size[0], size[1], file.projectionWkt(), file.geotransform());
    CHECK(warped.width == 8);
    CHECK(warped.height == 8);
    CHECK(warped.bounds3857.minX == Catch::Approx(12706639.339934358).epsilon(1e-6));
    CHECK(warped.bounds3857.minY == Catch::Approx(2544877.2370938584).epsilon(1e-6));
    CHECK(warped.bounds3857.maxX == Catch::Approx(12708803.936988916).epsilon(1e-6));
    CHECK(warped.bounds3857.maxY == Catch::Approx(2547041.834148416).epsilon(1e-6));
}

TEST_CASE("warping a very high-resolution raster caps the output size instead of growing unbounded") {
    // A real 43200x21600 global 30-arc-second WPS_GEOG dataset
    // (GMTED2010) warped at GDALWarp's natural ("preserve native
    // resolution") target size effectively hung - gigabytes and minutes,
    // not just slow. This raster's width (5000) alone already exceeds the
    // cap, without needing the full dataset's size or its extra
    // near-pole Mercator stretching, so the test stays fast.
    const Crs crs = Crs::lonLat();
    constexpr int nx = 5000, ny = 50;
    const std::array<double, 6> geotransform{-180.0, 360.0 / nx, 0.0, 25.0, 0.0, -50.0 / ny};
    const std::vector<float> values(static_cast<std::size_t>(nx) * ny, 1.0f);
    const auto warped = warpToWebMercator(values, nx, ny, crs.wkt(), geotransform);
    CHECK(warped.width <= 4096);
    CHECK(warped.height <= 4096);
    CHECK(warped.width == 4096);
}

TEST_CASE("raster layers render native WRF data with auto and manual ranges") {
    WrfSourceRegistry registry;
    auto& source = registry.open({"tests/fixtures/wrfout_multitime.nc"});
    const auto automatic = renderLayer(source, {.variable = "T2"});
    REQUIRE(automatic.pixels.size() == 64);
    CHECK(automatic.width == 8);
    CHECK(automatic.maximum > automatic.minimum);
    CHECK(automatic.bounds3857.maxX > automatic.bounds3857.minX);
    CHECK(automatic.bounds3857.maxY > automatic.bounds3857.minY);
    const auto manual = renderLayer(source, {.variable = "T2", .minimum = 270.0f, .maximum = 310.0f, .unitKey = "native"});
    CHECK(manual.minimum == 270.0f);
    CHECK(manual.maximum == 310.0f);
    CHECK(manual.pixels != automatic.pixels);
}

TEST_CASE("LayerRenderer caches slices/images and matches uncached rendering") {
    const std::string path = "tests/fixtures/wrfout_multitime.nc";
    LayerRenderer renderer;
    static_cast<void>(renderer.openFile({path}));
    const RasterLayer layer{.variable = "T2", .timeIndex = 1};
    const auto cached = renderer.render(path, layer);

    WrfSourceRegistry registry;
    auto& source = registry.open({path});
    const auto uncached = renderLayer(source, layer);
    CHECK(cached.pixels == uncached.pixels);
    CHECK(cached.minimum == uncached.minimum);
    CHECK(cached.bounds3857.minX == uncached.bounds3857.minX);

    // A second render() with identical settings is an image-cache hit -
    // same result, not a rebuild.
    const auto again = renderer.render(path, layer);
    CHECK(again.pixels == cached.pixels);

    // Different display units on the SAME (file, variable, time, level)
    // share one cached warped slice - converting for one layer must not
    // corrupt what a second layer in a different unit reads from it.
    const auto celsius = renderer.render(path, RasterLayer{.variable = "T2", .timeIndex = 1, .unitKey = "degC"});
    const auto kelvin = renderer.render(path, RasterLayer{.variable = "T2", .timeIndex = 1, .unitKey = "native"});
    CHECK(celsius.minimum != kelvin.minimum);  // different units, not accidentally sharing converted state
    const auto kelvinAgain = renderer.render(path, RasterLayer{.variable = "T2", .timeIndex = 1, .unitKey = "native"});
    CHECK(kelvinAgain.minimum == kelvin.minimum);  // native reading still intact after the degC render

    // invalidateFile drops the cache; the file can still be reopened and
    // rendered afterward (not left in a broken state).
    renderer.invalidateFile(path);
    CHECK(renderer.openPaths().empty());
    static_cast<void>(renderer.openFile({path}));
    const auto afterInvalidate = renderer.render(path, layer);
    CHECK(afterInvalidate.pixels == uncached.pixels);
}

TEST_CASE("valueAt rejects a latitude beyond Web Mercator's valid range instead of clamping into the raster") {
    LayerRenderer renderer;
    static_cast<void>(renderer.openFile({"tests/fixtures/geo_em_small.nc"}));
    const RasterLayer layer{.variable = "HGT_M"};
    // geo_em_small.nc's raster covers roughly lon 114.14..114.17, lat
    // 22.27..22.30 (near Hong Kong) - a normal in-range point right at the
    // center should resolve to a real value.
    CHECK(renderer.valueAt("tests/fixtures/geo_em_small.nc", layer, LonLat{114.1554, 22.2865}).has_value());
    // A cursor parked past the pole (e.g. after panning the map far beyond
    // its top/bottom edge) must read as "no data", not silently fold back
    // onto whatever the raster happens to hold at its own +-85.05 edge.
    CHECK_FALSE(renderer.valueAt("tests/fixtures/geo_em_small.nc", layer, LonLat{114.1554, 89.9}).has_value());
    CHECK_FALSE(renderer.valueAt("tests/fixtures/geo_em_small.nc", layer, LonLat{114.1554, -89.9}).has_value());
}

// --- LayerRenderer cache-tier hit/miss stats --------------------------------
// Ports test_rasterlayer.py's cache-behavior cases against LayerRenderer's
// stats() counters. Two Python cases have no C++ equivalent by design, not
// omission: overlay_for/effective_range/categorical_legend "of an unopened
// file returns None" - LayerRenderer::render() opens filePath on demand
// (WrfSourceRegistry::open is idempotent), it never distinguishes "not yet
// opened" from "open it now". RasterLayer.interpolate is likewise not part
// of RenderedRaster - it's applied by ViewForm when building the
// TileMapWidget overlay, one layer below LayerRenderer itself.

TEST_CASE("re-rendering the same layer hits the image cache") {
    LayerRenderer renderer;
    static_cast<void>(renderer.openFile({"tests/fixtures/geo_em_small.nc"}));
    const RasterLayer layer{.variable = "HGT_M"};
    static_cast<void>(renderer.render("tests/fixtures/geo_em_small.nc", layer));
    static_cast<void>(renderer.render("tests/fixtures/geo_em_small.nc", layer));
    CHECK(renderer.stats().sliceMisses == 1);
    CHECK(renderer.stats().sliceHits == 0);
    CHECK(renderer.stats().imageMisses == 1);
    CHECK(renderer.stats().imageHits == 1);
}

TEST_CASE("opacity change causes no new reads or images") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/geo_em_small.nc";
    static_cast<void>(renderer.openFile({path}));
    RasterLayer layer{.variable = "HGT_M", .opacity = 0.5};
    static_cast<void>(renderer.render(path, layer));
    const auto before = renderer.stats();
    layer.opacity = 0.1;
    static_cast<void>(renderer.render(path, layer));
    CHECK(renderer.stats().sliceMisses == before.sliceMisses);
    CHECK(renderer.stats().sliceHits == before.sliceHits);
    CHECK(renderer.stats().imageMisses == before.imageMisses);
    CHECK(renderer.stats().imageHits == before.imageHits + 1);
}

TEST_CASE("colormap change misses the image cache only") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/geo_em_small.nc";
    static_cast<void>(renderer.openFile({path}));
    RasterLayer layer{.variable = "HGT_M", .colormap = "viridis"};
    static_cast<void>(renderer.render(path, layer));
    const auto before = renderer.stats();
    layer.colormap = "plasma";
    static_cast<void>(renderer.render(path, layer));
    CHECK(renderer.stats().sliceMisses == before.sliceMisses);
    CHECK(renderer.stats().sliceHits == before.sliceHits + 1);
    CHECK(renderer.stats().imageMisses == before.imageMisses + 1);
}

TEST_CASE("time change misses both cache tiers") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/wrfout_multitime.nc";
    static_cast<void>(renderer.openFile({path}));
    RasterLayer layer{.variable = "T2", .timeIndex = 0};
    static_cast<void>(renderer.render(path, layer));
    const auto before = renderer.stats();
    layer.timeIndex = 1;
    static_cast<void>(renderer.render(path, layer));
    CHECK(renderer.stats().sliceMisses == before.sliceMisses + 1);
    CHECK(renderer.stats().imageMisses == before.imageMisses + 1);
}

TEST_CASE("stepping back in time is an image-cache hit") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/wrfout_multitime.nc";
    static_cast<void>(renderer.openFile({path}));
    RasterLayer layer{.variable = "T2", .timeIndex = 0};
    static_cast<void>(renderer.render(path, layer));
    layer.timeIndex = 1;
    static_cast<void>(renderer.render(path, layer));
    layer.timeIndex = 0;  // both its slice and image are still cached
    const auto before = renderer.stats();
    static_cast<void>(renderer.render(path, layer));
    CHECK(renderer.stats().sliceMisses == before.sliceMisses);
    CHECK(renderer.stats().sliceHits == before.sliceHits);
    CHECK(renderer.stats().imageHits == before.imageHits + 1);
}

TEST_CASE("prefetch populates the slice cache without building an image") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/wrfout_multitime.nc";
    static_cast<void>(renderer.openFile({path}));
    const RasterLayer layer{.variable = "T2", .timeIndex = 1};
    renderer.prefetch(path, layer);
    CHECK(renderer.stats().sliceMisses == 1);
    CHECK(renderer.stats().imageMisses == 0);
    const auto before = renderer.stats();
    static_cast<void>(renderer.render(path, layer));
    CHECK(renderer.stats().sliceMisses == before.sliceMisses);
    CHECK(renderer.stats().sliceHits == before.sliceHits + 1);
}

TEST_CASE("two layers on the same slice share the warp") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/geo_em_small.nc";
    static_cast<void>(renderer.openFile({path}));
    static_cast<void>(renderer.render(path, RasterLayer{.variable = "HGT_M", .colormap = "viridis"}));
    static_cast<void>(renderer.render(path, RasterLayer{.variable = "HGT_M", .colormap = "viridis"}));
    CHECK(renderer.stats().sliceMisses == 1);
}

TEST_CASE("invalidating a file does not affect another open file's cache") {
    LayerRenderer renderer;
    static_cast<void>(renderer.openFile({"tests/fixtures/geo_em_small.nc"}));
    static_cast<void>(renderer.openFile({"tests/fixtures/wrfout_multitime.nc"}));
    static_cast<void>(renderer.render("tests/fixtures/geo_em_small.nc", RasterLayer{.variable = "HGT_M"}));
    const RasterLayer wrfoutLayer{.variable = "T2"};
    static_cast<void>(renderer.render("tests/fixtures/wrfout_multitime.nc", wrfoutLayer));
    renderer.invalidateFile("tests/fixtures/geo_em_small.nc");
    const auto before = renderer.stats();
    static_cast<void>(renderer.render("tests/fixtures/wrfout_multitime.nc", wrfoutLayer));
    CHECK(renderer.stats().imageHits == before.imageHits + 1);  // untouched by the other file's invalidation
}

TEST_CASE("clear resets the cache stats") {
    LayerRenderer renderer;
    static_cast<void>(renderer.openFile({"tests/fixtures/geo_em_small.nc"}));
    static_cast<void>(renderer.render("tests/fixtures/geo_em_small.nc", RasterLayer{.variable = "HGT_M"}));
    renderer.clear();
    CHECK(renderer.stats().sliceMisses == 0);
    CHECK(renderer.stats().imageMisses == 0);
}

TEST_CASE("effective range honors a manual override and is auto otherwise") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/geo_em_small.nc";
    static_cast<void>(renderer.openFile({path}));
    const auto autoRange = renderer.render(path, RasterLayer{.variable = "HGT_M"});
    CHECK(autoRange.minimum < autoRange.maximum);

    const auto manual = renderer.render(path, RasterLayer{.variable = "HGT_M", .minimum = 10.0f, .maximum = 200.0f});
    CHECK(manual.minimum == Catch::Approx(10.0));
    CHECK(manual.maximum == Catch::Approx(200.0));
}

TEST_CASE("effective range in a converted unit is shifted from native") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/wrfout_multitime.nc";
    static_cast<void>(renderer.openFile({path}));
    const auto native = renderer.render(path, RasterLayer{.variable = "T2"});
    const auto celsius = renderer.render(path, RasterLayer{.variable = "T2", .unitKey = "degC"});
    CHECK(celsius.minimum == Catch::Approx(native.minimum - 273.15).epsilon(1e-3));
    CHECK(celsius.maximum == Catch::Approx(native.maximum - 273.15).epsilon(1e-3));
}

TEST_CASE("manual range in a converted unit is used as-is") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/wrfout_multitime.nc";
    static_cast<void>(renderer.openFile({path}));
    const auto rendered = renderer.render(path, RasterLayer{.variable = "T2", .minimum = 0.0f, .maximum = 30.0f, .unitKey = "degC"});
    CHECK(rendered.minimum == Catch::Approx(0.0));
    CHECK(rendered.maximum == Catch::Approx(30.0));
}

TEST_CASE("units change misses the image cache but not the slice cache") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/wrfout_multitime.nc";
    static_cast<void>(renderer.openFile({path}));
    RasterLayer layer{.variable = "T2"};
    static_cast<void>(renderer.render(path, layer));
    const auto before = renderer.stats();
    layer.unitKey = "degC";
    static_cast<void>(renderer.render(path, layer));
    CHECK(renderer.stats().sliceMisses == before.sliceMisses);
    CHECK(renderer.stats().sliceHits == before.sliceHits + 1);
    CHECK(renderer.stats().imageMisses == before.imageMisses + 1);
}

TEST_CASE("two layers with the same slice but different units do not share the image cache") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/wrfout_multitime.nc";
    static_cast<void>(renderer.openFile({path}));
    static_cast<void>(renderer.render(path, RasterLayer{.variable = "T2"}));
    static_cast<void>(renderer.render(path, RasterLayer{.variable = "T2", .unitKey = "degC"}));
    CHECK(renderer.stats().sliceMisses == 1);
    CHECK(renderer.stats().imageMisses == 2);
}

TEST_CASE("tick settings do not invalidate either cache tier") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/geo_em_small.nc";
    static_cast<void>(renderer.openFile({path}));
    RasterLayer layer{.variable = "HGT_M"};
    static_cast<void>(renderer.render(path, layer));
    const auto before = renderer.stats();
    layer.tickCount = 9; layer.tickFormat = "scientific"; layer.tickDecimals = 4;
    static_cast<void>(renderer.render(path, layer));
    CHECK(renderer.stats().sliceMisses == before.sliceMisses);
    CHECK(renderer.stats().sliceHits == before.sliceHits);
    CHECK(renderer.stats().imageMisses == before.imageMisses);
    CHECK(renderer.stats().imageHits == before.imageHits + 1);
}

TEST_CASE("LayerRenderer.openFile with a series registers one entry and dedupes on reopen") {
    LayerRenderer renderer;
    const std::vector<std::filesystem::path> seriesPaths{
        "tests/fixtures/wrfout_d01_2020-01-01_00_00_00.nc", "tests/fixtures/wrfout_d01_2020-01-01_00_30_00.nc",
        "tests/fixtures/wrfout_d01_2020-01-01_01_00_00.nc"};
    auto& first = renderer.openFile(seriesPaths);
    CHECK(renderer.openPaths() == std::vector<std::string>{seriesPaths.front().string()});
    auto& second = renderer.openFile(seriesPaths);
    CHECK(&first == &second);  // reopening the same series is a registry hit, not a rebuild
}

TEST_CASE("render works at every time index of a series") {
    LayerRenderer renderer;
    const std::vector<std::filesystem::path> seriesPaths{
        "tests/fixtures/wrfout_d01_2020-01-01_00_00_00.nc", "tests/fixtures/wrfout_d01_2020-01-01_00_30_00.nc",
        "tests/fixtures/wrfout_d01_2020-01-01_01_00_00.nc"};
    auto& source = renderer.openFile(seriesPaths);
    const auto times = source.seriesTimes();
    REQUIRE(times);
    for (int timeIndex = 0; timeIndex < static_cast<int>(times->size()); ++timeIndex) {
        const auto rendered = renderer.render(seriesPaths.front().string(), RasterLayer{.variable = "T2", .timeIndex = timeIndex});
        CHECK(rendered.width > 0);
        CHECK(rendered.height > 0);
    }
}

TEST_CASE("slice cache evicts by total bytes, not entry count") {
    LayerRenderer renderer(/*sliceCacheBytes=*/1, kDefaultImageCacheSize);
    const std::string path = "tests/fixtures/geo_em_small.nc";
    static_cast<void>(renderer.openFile({path}));
    static_cast<void>(renderer.render(path, RasterLayer{.variable = "HGT_M"}));
    static_cast<void>(renderer.render(path, RasterLayer{.variable = "LU_INDEX", .colormap = kCategoricalColormap}));
    CHECK(renderer.stats().sliceMisses == 2);
    // A different colormap forces an image-cache miss too, so this render
    // actually consults the slice tier instead of short-circuiting on an
    // image hit from the first render above - HGT_M's slice was evicted to
    // make room for LU_INDEX's, so this is a slice miss again.
    static_cast<void>(renderer.render(path, RasterLayer{.variable = "HGT_M", .colormap = "plasma"}));
    CHECK(renderer.stats().sliceMisses == 3);
}

TEST_CASE("image cache evicts by entry count") {
    LayerRenderer renderer(kDefaultSliceCacheBytes, /*imageCacheSize=*/1);
    const std::string path = "tests/fixtures/geo_em_small.nc";
    static_cast<void>(renderer.openFile({path}));
    static_cast<void>(renderer.render(path, RasterLayer{.variable = "HGT_M", .colormap = "viridis"}));
    static_cast<void>(renderer.render(path, RasterLayer{.variable = "HGT_M", .colormap = "plasma"}));
    CHECK(renderer.stats().imageMisses == 2);
    // viridis's image was evicted to make room - re-rendering it is a miss again.
    static_cast<void>(renderer.render(path, RasterLayer{.variable = "HGT_M", .colormap = "viridis"}));
    CHECK(renderer.stats().imageMisses == 3);
}

TEST_CASE("LayerRenderer.render exposes the file's own landuse scheme for a categorical layer") {
    // geo_em_small.nc's MMINLU is MODIFIED_IGBP_MODIS_NOAH; its LU_INDEX
    // slice contains exactly classes 2, 13, 17 (verified against the fixture).
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/geo_em_small.nc";
    static_cast<void>(renderer.openFile({path}));
    const auto rendered = renderer.render(path, RasterLayer{.variable = "LU_INDEX", .colormap = kCategoricalColormap});
    CHECK(rendered.presentCategories == std::vector<int>{2, 13, 17});
    CHECK(rendered.categoricalLabels.at(2) == "Evergreen Broadleaf Forest");
    CHECK(rendered.categoricalPalette[2] == Rgb{0x00, 0xFF, 0x00});
    CHECK(rendered.categoricalLabels.at(13) == "Urban and Built-Up");
    CHECK(rendered.categoricalPalette[13] == Rgb{0xFF, 0x00, 0x00});
    CHECK(rendered.categoricalLabels.at(17) == "Water");
    CHECK(rendered.categoricalPalette[17] == Rgb{0x00, 0x00, 0x80});
}

TEST_CASE("a continuous layer's render carries no categorical legend data") {
    LayerRenderer renderer;
    const std::string path = "tests/fixtures/geo_em_small.nc";
    static_cast<void>(renderer.openFile({path}));
    const auto rendered = renderer.render(path, RasterLayer{.variable = "HGT_M"});
    CHECK(rendered.presentCategories.empty());
    CHECK(rendered.categoricalLabels.empty());
}

TEST_CASE("isWpsGeogDataset recognizes a directory with an index file, not a plain file or directory") {
    CHECK(isWpsGeogDataset("tests/fixtures/geotiff_convert/wps_soiltemp_1deg"));
    CHECK_FALSE(isWpsGeogDataset("tests/fixtures/geotiff_convert/wps_soiltemp_1deg/index"));
    CHECK_FALSE(isWpsGeogDataset("tests/fixtures/geotiff_convert/utm.tif"));
    CHECK_FALSE(isWpsGeogDataset("tests/fixtures/does-not-exist"));
}

TEST_CASE("WpsBinarySource reads a real WPS_GEOG regular_ll dataset with the correct geometry") {
    // tests/fixtures/geotiff_convert/wps_soiltemp_1deg/index: regular_ll,
    // dx=dy=1.0 degree, known_x=known_y=1.0 (an integer -> WPS/GEOGRID's
    // cell-center convention) at known_lon=-179.5/known_lat=-89.5 - the
    // real center of the southwesternmost 1-degree cell of a global grid,
    // so the raster's true edges are exactly -180/-90 (west/south), not
    // -179.5/-89.5.
    WpsBinarySource source("tests/fixtures/geotiff_convert/wps_soiltemp_1deg");
    CHECK(source.size() == std::array<int, 2>{180, 180});
    CHECK(source.displayName() == "wps_soiltemp_1deg");
    const auto& gt = source.geotransform();
    CHECK(gt[0] == Catch::Approx(-180.0));
    CHECK(gt[1] == Catch::Approx(1.0));
    CHECK(gt[2] == Catch::Approx(0.0));
    CHECK(gt[3] == Catch::Approx(90.0));
    CHECK(gt[4] == Catch::Approx(0.0));
    CHECK(gt[5] == Catch::Approx(-1.0));
    REQUIRE(source.variables().size() == 1);
    const auto& variable = source.variables().front();
    CHECK(variable.name == "Annual mean deep soil temperature");
    CHECK(variable.units == "Kelvin");
    CHECK(variable.levelCount == 1);
    CHECK_FALSE(variable.categoryScheme.has_value());
    const auto values = source.read(variable.name, 0, 0);
    REQUIRE(values.size() == 180 * 180);
    // A row-1 (tile-file numbering, south) sample: dy is non-negative so
    // read() must have flipped it up to the last (northmost-index) output
    // row - CHECK it lands there, not at row 0.
    const bool anyFiniteInLastRow = std::any_of(values.end() - 180, values.end(), [](float v) { return std::isfinite(v); });
    CHECK(anyFiniteInLastRow);
}

TEST_CASE("WpsBinarySource re-wraps a global 0-360 longitude dataset into -180..180") {
    // A hand-built minimal global regular_ll dataset: 4 columns x 90
    // degrees = the full 360-degree circle, one row (row-flip is already
    // covered by the fixture-backed test above; this one is only about the
    // column wraparound). known_lon=0 at known_x=1 (cell-center
    // convention) puts raw column centers at 0, 90, 180, 270 degrees - a
    // real-world pattern real global WPS_GEOG datasets like GMTED2010 use
    // (0..360 rather than -180..180), which without unwrapping leaves the
    // >180 portion projected far outside web-Mercator's valid range
    // instead of wrapping back to the western hemisphere.
    const auto dir = std::filesystem::temp_directory_path() / "wrftools-cpp-wps-geog-wrap";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    {
        std::ofstream index(dir / "index");
        index << "type=continuous\n"
                 "projection=regular_ll\n"
                 "missing_value=255\n"
                 "dx=90.0\n"
                 "dy=90.0\n"
                 "known_x=1.0\n"
                 "known_y=1.0\n"
                 "known_lat=0.0\n"
                 "known_lon=0.0\n"
                 "wordsize=1\n"
                 "signed=no\n"
                 "endian=big\n"
                 "tile_x=4\n"
                 "tile_y=1\n"
                 "tile_z=1\n"
                 "tile_bdr=0\n"
                 "scale_factor=1\n"
                 "units=\"none\"\n"
                 "description=\"wrap test\"\n";
    }
    {
        // Single tile, no border: raw column i (0-based, longitude
        // i*90 degrees) holds value i, so the column reordering can be
        // checked directly against the output.
        std::ofstream tile(dir / "00001-00004.00001-00001", std::ios::binary);
        const unsigned char values[4] = {0, 1, 2, 3};
        tile.write(reinterpret_cast<const char*>(values), sizeof(values));
    }

    WpsBinarySource source(dir);
    CHECK(source.size() == std::array<int, 2>{4, 1});
    const auto& gt = source.geotransform();
    CHECK(gt[0] == Catch::Approx(-180.0));
    CHECK(gt[1] == Catch::Approx(90.0));
    const auto values = source.read(source.variables().front().name, 0, 0);
    REQUIRE(values.size() == 4);
    // Output columns represent -180, -90, 0, 90 degrees, i.e. raw columns
    // 2 (180=-180), 3 (270=-90), 0 (0), 1 (90).
    CHECK(values == std::vector<float>{2, 3, 0, 1});

    std::filesystem::remove_all(dir);
}

TEST_CASE("WpsBinarySource crops tile-alignment padding beyond the true global extent") {
    // Some real-world global regular_ll datasets pad nx/ny out to a whole
    // number of tiles beyond the true 360deg/180deg extent (observed in
    // NCAR's own topo_gmted2010_5m: tile_x=600 doesn't divide the true
    // 4320 columns evenly, so it ships 4800 - the trailing 480 columns are
    // a literal duplicate of columns 0..479, and 240 trailing rows are
    // zero-filled past the pole). A hand-built minimal case of the same
    // shape: dx=dy=90 (true size 4x2), raw tile is 6 columns x 3 rows -
    // 2 extra padding columns and 1 extra padding row, filled with
    // sentinel values (44/55/144/155/200) that must never appear in the
    // output. The 4 real columns still span the full 360 degrees, so this
    // also exercises the column-wraparound fix on top of the crop.
    const auto dir = std::filesystem::temp_directory_path() / "wrftools-cpp-wps-geog-pad";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    {
        std::ofstream index(dir / "index");
        index << "type=continuous\n"
                 "projection=regular_ll\n"
                 "missing_value=255\n"
                 "dx=90.0\n"
                 "dy=90.0\n"
                 "known_x=1.0\n"
                 "known_y=1.0\n"
                 "known_lat=-45.0\n"
                 "known_lon=0.0\n"
                 "wordsize=1\n"
                 "signed=no\n"
                 "endian=big\n"
                 "tile_x=6\n"
                 "tile_y=3\n"
                 "tile_z=1\n"
                 "tile_bdr=0\n"
                 "scale_factor=1\n"
                 "units=\"none\"\n"
                 "description=\"pad test\"\n";
    }
    {
        std::ofstream tile(dir / "00001-00006.00001-00003", std::ios::binary);
        // Row-major, tile-file row order: row 0 (south, real), row 1
        // (north, real), row 2 (padding, must be dropped entirely).
        const unsigned char values[18] = {
            0, 1, 2, 3, 44, 55,       // south: real cols 0-3, padding cols 4-5
            10, 11, 12, 13, 144, 155, // north: real cols 0-3, padding cols 4-5
            200, 200, 200, 200, 200, 200,  // padding row: must never appear
        };
        tile.write(reinterpret_cast<const char*>(values), sizeof(values));
    }

    WpsBinarySource source(dir);
    CHECK(source.size() == std::array<int, 2>{4, 2});
    const auto& gt = source.geotransform();
    CHECK(gt[0] == Catch::Approx(-180.0));
    CHECK(gt[3] == Catch::Approx(90.0));
    const auto values = source.read(source.variables().front().name, 0, 0);
    REQUIRE(values.size() == 8);
    // Row 0 = north (raw row 1: 10,11,12,13), row 1 = south (raw row 0:
    // 0,1,2,3), each column-wrapped the same way as the plain wraparound
    // test above (known_lon=0/known_x=1/dx=90 -> shift=2). No 44/55/144/
    // 155/200 sentinel may appear anywhere.
    CHECK(values == std::vector<float>{12, 13, 10, 11, 2, 3, 0, 1});

    std::filesystem::remove_all(dir);
}

TEST_CASE("WpsBinarySource matches geogrid.exe's own defaults for absent tile_bdr/missing_value keys") {
    // Cross-checked against WPS geogrid's own source
    // (source_data_module.f90's get_tile_dimensions/get_missing_value):
    // an absent tile_bdr defaults to 0 (not this tool's own -b 3 CLI
    // default - a wrong default here would make read_tiles expect a much
    // larger tile file than this test provides and fail outright), and an
    // absent missing_value means "no value is missing" (not 0 - which
    // would wrongly mask the real 0 value this test includes at south/
    // col0).
    const auto dir = std::filesystem::temp_directory_path() / "wrftools-cpp-wps-geog-defaults";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    {
        std::ofstream index(dir / "index");
        index << "type=continuous\n"
                 "projection=regular_ll\n"
                 "dx=90.0\n"
                 "dy=90.0\n"
                 "known_x=1.0\n"
                 "known_y=1.0\n"
                 "known_lat=-45.0\n"
                 "known_lon=-135.0\n"
                 "wordsize=1\n"
                 "signed=no\n"
                 "endian=big\n"
                 "tile_x=4\n"
                 "tile_y=2\n"
                 "tile_z=1\n"
                 "scale_factor=1\n"
                 "units=\"none\"\n"
                 "description=\"defaults test\"\n";
        // No tile_bdr, no missing_value, no row_order key.
    }
    {
        // Exactly tile_x*tile_y bytes - only valid if tile_bdr correctly
        // defaults to 0; the old (wrong) default of 3 would need
        // (4+6)*(2+6)=80 bytes and fail to read this 8-byte file.
        std::ofstream tile(dir / "00001-00004.00001-00002", std::ios::binary);
        const unsigned char values[8] = {0, 1, 2, 3, 10, 11, 12, 13};
        tile.write(reinterpret_cast<const char*>(values), sizeof(values));
    }

    WpsBinarySource source(dir);
    CHECK(source.size() == std::array<int, 2>{4, 2});
    const auto& gt = source.geotransform();
    CHECK(gt[0] == Catch::Approx(-180.0));  // known_lon already centers col 0 on -135, no wraparound needed
    const auto values = source.read(source.variables().front().name, 0, 0);
    REQUIRE(values.size() == 8);
    // No row_order key -> geogrid.exe's own bottom_top default -> row 1
    // (south, values 0-3) flips to the second (southern) output row.
    CHECK(values == std::vector<float>{10, 11, 12, 13, 0, 1, 2, 3});

    std::filesystem::remove_all(dir);
}

TEST_CASE("WpsBinarySource honors an explicit row_order=top_bottom key over dy's sign") {
    // Same 4x2 dataset as above, but with row_order=top_bottom explicitly
    // set while dy is still positive - geogrid.exe reads row_order as its
    // own key (get_row_order), unrelated to dy's sign, so this must NOT
    // flip (row 1 is already north here), unlike the bottom_top-default
    // case above which does flip with the exact same positive dy.
    const auto dir = std::filesystem::temp_directory_path() / "wrftools-cpp-wps-geog-toptobottom";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    {
        std::ofstream index(dir / "index");
        index << "type=continuous\n"
                 "projection=regular_ll\n"
                 "row_order=top_bottom\n"
                 "missing_value=255\n"
                 "dx=90.0\n"
                 "dy=90.0\n"
                 "known_x=1.0\n"
                 "known_y=1.0\n"
                 "known_lat=-45.0\n"
                 "known_lon=-135.0\n"
                 "wordsize=1\n"
                 "signed=no\n"
                 "endian=big\n"
                 "tile_x=4\n"
                 "tile_y=2\n"
                 "tile_z=1\n"
                 "tile_bdr=0\n"
                 "scale_factor=1\n"
                 "units=\"none\"\n"
                 "description=\"top_bottom test\"\n";
    }
    {
        std::ofstream tile(dir / "00001-00004.00001-00002", std::ios::binary);
        const unsigned char values[8] = {0, 1, 2, 3, 10, 11, 12, 13};
        tile.write(reinterpret_cast<const char*>(values), sizeof(values));
    }

    WpsBinarySource source(dir);
    const auto values = source.read(source.variables().front().name, 0, 0);
    REQUIRE(values.size() == 8);
    // Unflipped: tile-file row 1 (values 0-3) is already the north (first
    // output) row.
    CHECK(values == std::vector<float>{0, 1, 2, 3, 10, 11, 12, 13});

    std::filesystem::remove_all(dir);
}

TEST_CASE("WrfSourceRegistry opens a WPS_GEOG directory as a WpsBinarySource, not a NetCDF file") {
    WrfSourceRegistry registry;
    auto& source = registry.open({"tests/fixtures/geotiff_convert/wps_soiltemp_1deg"});
    CHECK(source.variables().size() == 1);
    CHECK(source.displayName() == "wps_soiltemp_1deg");
}

namespace {
std::filesystem::path lczFixtureCopy(const std::string& suffix) {
    // temp_directory_path(), not a literal "build" - matches this file's
    // own established convention elsewhere (the WPS_GEOG scratch-dir
    // tests above) rather than assuming a directory literally named
    // "build" exists relative to the test process's working directory:
    // CI's actual CMake binary dir is "build-cpp", not "build" (see
    // .github/workflows/cpp.yml) - a bare "build" literal here silently
    // depended on whichever binary dir name a developer happened to use
    // locally, and failed outright in CI on all three platforms with a
    // misleading "No such file or directory" (looks like a missing SOURCE
    // fixture; it's actually the missing DESTINATION directory).
    const auto dst = std::filesystem::temp_directory_path() / ("wrftools-cpp-netcdf-file-test-" + suffix + ".nc");
    NetcdfFile::copyFile("tests/fixtures/lcz/5by5.nc", dst);
    return dst;
}
}  // namespace

TEST_CASE("NetcdfFile reads dimensions, variables, and attributes from a real geo_em file") {
    const auto file = NetcdfFile::open("tests/fixtures/lcz/5by5.nc", NetcdfFile::Mode::ReadOnly);

    const auto dims = file.dimensions();
    const auto southNorth = std::find_if(dims.begin(), dims.end(), [](const auto& d) { return d.name == "south_north"; });
    REQUIRE(southNorth != dims.end());
    CHECK(southNorth->length == 5);
    CHECK_FALSE(southNorth->isUnlimited);

    const auto time = std::find_if(dims.begin(), dims.end(), [](const auto& d) { return d.name == "Time"; });
    REQUIRE(time != dims.end());
    CHECK(time->isUnlimited);

    REQUIRE(file.hasVariable("LU_INDEX"));
    CHECK_FALSE(file.hasVariable("NOT_A_REAL_VARIABLE"));
    const auto luIndex = file.variable("LU_INDEX");
    CHECK(luIndex.dimensionNames == std::vector<std::string>{"Time", "south_north", "west_east"});
    CHECK(file.shape("LU_INDEX") == std::vector<std::size_t>{1, 5, 5});

    CHECK(file.hasAttribute("", "NUM_LAND_CAT"));
    const auto numLandCat = file.getAttribute("", "NUM_LAND_CAT");
    REQUIRE(numLandCat.numbers.size() == 1);
    CHECK(numLandCat.numbers[0] == 41);

    CHECK(file.hasAttribute("LU_INDEX", "description"));
    CHECK(file.getAttribute("LU_INDEX", "description").text == "Dominant category");

    const auto luValues = file.readFloat("LU_INDEX");
    CHECK(luValues.size() == 25);
}

TEST_CASE("NetcdfFile::copyFile round-trips a real file byte-for-byte") {
    const auto dst = lczFixtureCopy("copy");
    REQUIRE(std::filesystem::exists(dst));
    CHECK(std::filesystem::file_size(dst) == std::filesystem::file_size("tests/fixtures/lcz/5by5.nc"));

    {
        // Scoped so both ifstreams close before the removal below - an
        // open handle onto `dst` (even a read-only one) makes
        // std::filesystem::remove fail on Windows, unlike POSIX where an
        // open file can be unlinked freely.
        std::ifstream original("tests/fixtures/lcz/5by5.nc", std::ios::binary);
        std::ifstream copy(dst, std::ios::binary);
        const std::vector<char> originalBytes((std::istreambuf_iterator<char>(original)), std::istreambuf_iterator<char>());
        const std::vector<char> copyBytes((std::istreambuf_iterator<char>(copy)), std::istreambuf_iterator<char>());
        CHECK(originalBytes == copyBytes);
    }

    std::filesystem::remove(dst);
}

TEST_CASE("NetcdfFile mutates a variable and a global attribute without disturbing the rest of the file") {
    const auto dst = lczFixtureCopy("mutate");
    {
        auto file = NetcdfFile::open(dst, NetcdfFile::Mode::ReadWrite);
        auto luIndex = file.readFloat("LU_INDEX");
        std::fill(luIndex.begin(), luIndex.end(), 13.0f);
        file.writeFloat("LU_INDEX", luIndex);

        auto attribute = file.getAttribute("", "NUM_LAND_CAT");
        attribute.numbers[0] = 21;
        file.putAttribute("", attribute);
    }

    {
        // Scoped so `reopened` closes before the removal below - see the
        // copyFile test above's comment on why an open handle blocks it
        // on Windows.
        const auto reopened = NetcdfFile::open(dst, NetcdfFile::Mode::ReadOnly);
        const auto luIndex = reopened.readFloat("LU_INDEX");
        CHECK(std::all_of(luIndex.begin(), luIndex.end(), [](float v) { return v == 13.0f; }));
        CHECK(reopened.getAttribute("", "NUM_LAND_CAT").numbers[0] == 21);

        // Untouched variable/attribute survive the mutation unchanged.
        CHECK(reopened.readFloat("XLAT_M").size() == 25);
        CHECK(reopened.getAttribute("LU_INDEX", "description").text == "Dominant category");
    }

    std::filesystem::remove(dst);
}

TEST_CASE("wrfGridInfo matches w2w's own _get_wrf_grid_info for a MAP_PROJ=6 (lat-lon/eqc) domain") {
    const auto file = NetcdfFile::open("tests/fixtures/lcz/5by5.nc", NetcdfFile::Mode::ReadOnly);
    const auto info = wrfGridInfo(file);

    CHECK(info.width == 5);
    CHECK(info.height == 5);
    // Pinned against `uv run python -c "import w2w.w2w as w; ..."` on
    // w2w's own add_wrf_version branch (_get_wrf_grid_info is unchanged
    // there vs. main) against this exact fixture.
    CHECK(info.geotransform[0] == Catch::Approx(-751560.7950914681));
    CHECK(info.geotransform[1] == Catch::Approx(1111.7747802734375));
    CHECK(info.geotransform[2] == Catch::Approx(0.0));
    CHECK(info.geotransform[3] == Catch::Approx(4631097.686290965));
    CHECK(info.geotransform[4] == Catch::Approx(0.0));
    CHECK(info.geotransform[5] == Catch::Approx(1111.7747802734375));  // positive - see WrfGridInfo's doc comment
}

TEST_CASE("wrfGridInfo matches w2w's own _get_wrf_grid_info for a MAP_PROJ=1 (Lambert) domain") {
    const auto file = NetcdfFile::open("tests/fixtures/lcz/geo_em.d01_Shanghai_ncl20.nc", NetcdfFile::Mode::ReadOnly);
    const auto info = wrfGridInfo(file);

    CHECK(info.width == 90);
    CHECK(info.height == 90);
    CHECK(info.geotransform[0] == Catch::Approx(-44998.556668209276));
    CHECK(info.geotransform[1] == Catch::Approx(1000.0));
    CHECK(info.geotransform[2] == Catch::Approx(0.0));
    CHECK(info.geotransform[3] == Catch::Approx(-44999.156109511205));
    CHECK(info.geotransform[4] == Catch::Approx(0.0));
    CHECK(info.geotransform[5] == Catch::Approx(1000.0));
}

namespace {
// Round-trips a handful of wrfGridInfo-derived pixel centers back to lon/
// lat and compares them against the file's own geogrid.exe-written
// XLAT_M/XLONG_M arrays at those same indices - the ground-truth check for
// "is this grid actually where geogrid put it", not just "internally
// self-consistent" or "matches w2w.py's own arithmetic" (both already
// covered by the pinned-geotransform tests above). maxErrorMeters is a
// loose bound (real observed error on these fixtures is 1-3m, against
// 1000-1112m pixel spacing) - this is a regression guard against a real
// mispositioning bug reappearing, not a tight numerical pin.
void checkGridMatchesFileCoordinates(const std::filesystem::path& path, double maxErrorMeters) {
    const auto file = NetcdfFile::open(path, NetcdfFile::Mode::ReadOnly);
    const auto grid = wrfGridInfo(file);
    const auto xlat = file.readFloat("XLAT_M");
    const auto xlong = file.readFloat("XLONG_M");
    const auto shape = file.shape("XLAT_M");  // {1, ny, nx}
    const int nx = static_cast<int>(shape[2]), ny = static_cast<int>(shape[1]);

    const int checks[][2] = {{0, 0}, {nx - 1, 0}, {0, ny - 1}, {nx - 1, ny - 1}, {nx / 2, ny / 2}};
    for (const auto& rc : checks) {
        const int col = rc[0], row = rc[1];
        const double px = grid.geotransform[0] + (col + 0.5) * grid.geotransform[1] + (row + 0.5) * grid.geotransform[2];
        const double py = grid.geotransform[3] + (col + 0.5) * grid.geotransform[4] + (row + 0.5) * grid.geotransform[5];
        const auto lonLat = grid.crs.toLonLat({px, py});
        const std::size_t idx = static_cast<std::size_t>(row) * static_cast<std::size_t>(nx) + static_cast<std::size_t>(col);
        const double dLat = lonLat.lat - xlat[idx], dLon = lonLat.lon - xlong[idx];
        const double errorMeters = std::sqrt(dLat * dLat + dLon * dLon) * 111000.0;  // rough deg->m, plenty for this bound
        CAPTURE(path.string(), col, row, lonLat.lon, lonLat.lat, xlong[idx], xlat[idx], errorMeters);
        CHECK(errorMeters < maxErrorMeters);
    }
}
}  // namespace

TEST_CASE("wrfGridInfo's grid lines up with geogrid.exe's own XLAT_M/XLONG_M coordinates, not just w2w.py's own arithmetic") {
    // The pinned-geotransform tests above only prove this port's numbers
    // match w2w.py's own numbers - not that either is geographically
    // correct. This checks against real geogrid.exe output directly: a
    // real Lambert domain (Shanghai) and a real lat-lon/eqc domain
    // (Zaragoza), corner and center pixels, well under one pixel of
    // agreement in every case.
    checkGridMatchesFileCoordinates("tests/fixtures/lcz/geo_em.d02_Shanghai.nc", 5.0);  // MAP_PROJ=1, dx=dy=1000m
    checkGridMatchesFileCoordinates("tests/fixtures/lcz/geo_em.d04.nc", 5.0);            // MAP_PROJ=6, dx=dy=1111.8m
    // Mercator (MAP_PROJ=3) has no real geogrid.exe-produced fixture with
    // self-consistent CEN_LON/CEN_LAT in this repo to pin against here
    // (tests/fixtures/geo_em_small.nc is a hand-crafted fixture whose own
    // CEN_LON/CEN_LAT/STAND_LON don't match its own XLAT_M/XLONG_M arrays
    // - confirmed by inspection, not a wrfGridInfo bug) - separately
    // verified by hand against a real user-supplied Hong Kong domain
    // (three real Mercator geo_em files, sub-1.4m agreement on 250-6250m
    // grids), see PORT_W2W.MD.
}

TEST_CASE("warpToGrid reprojects into a caller-specified destination grid, not just EPSG:3857") {
    const auto sourceCrs = Crs::fromProj4("+proj=merc +lon_0=0 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +no_defs");
    const int width = 4, height = 4;
    std::vector<float> values(static_cast<std::size_t>(width) * height);
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) values[static_cast<std::size_t>(row) * width + col] = static_cast<float>(col);
    const std::array<double, 6> sourceGt{0.0, 100.0, 0.0, 400.0, 0.0, -100.0};  // north-up, dx=100, origin (0,400)

    // Same CRS, same extent, half the resolution: each destination pixel
    // should average a 2x2 block of the source.
    const std::array<double, 6> destGt{0.0, 200.0, 0.0, 400.0, 0.0, -200.0};
    const auto warped = warpToGrid(values, width, height, sourceCrs.wkt(), sourceGt, sourceCrs.wkt(), destGt, 2, 2, ResampleMethod::Average);
    REQUIRE(warped.size() == 4);
    CHECK(warped[0] == Catch::Approx(0.5).margin(0.05));  // top-left: source cols {0,1}
    CHECK(warped[1] == Catch::Approx(2.5).margin(0.05));  // top-right: source cols {2,3}
    CHECK(warped[2] == Catch::Approx(0.5).margin(0.05));
    CHECK(warped[3] == Catch::Approx(2.5).margin(0.05));
}

TEST_CASE("removeUrban matches a live w2w.wrf_remove_urban run (add_wrf_version) on a real 5x5 domain") {
    const auto dst = lczFixtureCopy("remove_urban_src");
    const auto out = std::filesystem::temp_directory_path() / "wrftools-cpp-netcdf-file-test-remove-urban-out.nc";
    std::filesystem::remove(out);

    // Matches `m.wrf_remove_urban(info, NPIX_NLC=3, NPIX_AREA=9)` run
    // against this exact fixture (tests/fixtures/lcz/5by5.nc, copied from
    // w2w's own testing/) on the add_wrf_version branch at 7801b3e.
    removeUrban(dst, out, 3, 9);

    // Scoped so `result` closes before removing `out` below - an open
    // handle onto it blocks std::filesystem::remove on Windows.
    {
        const auto result = NetcdfFile::open(out, NetcdfFile::Mode::ReadOnly);
        CHECK(result.getAttribute("", "NUM_LAND_CAT").numbers[0] == 21);

        const std::vector<float> expectedLu{
            7, 12, 7, 7, 14,
            12, 12, 12, 12, 12,
            12, 12, 12, 12, 12,
            12, 5, 12, 12, 12,
            12, 12, 12, 12, 14,
        };
        const auto lu = result.readFloat("LU_INDEX");
        REQUIRE(lu.size() == expectedLu.size());
        for (std::size_t i = 0; i < lu.size(); ++i) CHECK(lu[i] == expectedLu[i]);

        constexpr std::size_t ny = 5, nx = 5, npix = ny * nx;
        const auto greenf = result.readFloat("GREENFRAC");
        const std::vector<float> expectedGreenMonth0{
            0.32288238f, 0.34568438f, 0.31409433f, 0.30929843f, 0.27927542f,
            0.35764524f, 0.40199462f, 0.36666667f, 0.28f, 0.25155333f,
            0.29333332f, 0.3633333f, 0.36111107f, 0.3127196f, 0.29627478f,
            0.28f, 0.32999998f, 0.35333332f, 0.36188415f, 0.3021261f,
            0.29270646f, 0.30135322f, 0.31f, 0.3196875f, 0.32455063f,
        };
        const std::vector<float> expectedGreenMonth6{
            0.34078887f, 0.41630322f, 0.38273245f, 0.37449577f, 0.31640434f,
            0.38326782f, 0.47702634f, 0.45f, 0.29f, 0.20068237f,
            0.3333333f, 0.5f, 0.49111113f, 0.37584224f, 0.2759695f,
            0.28666666f, 0.42f, 0.5233333f, 0.56155723f, 0.46738f,
            0.34370252f, 0.38851792f, 0.4333333f, 0.45120147f, 0.49080566f,
        };
        for (std::size_t p = 0; p < npix; ++p) {
            CHECK(greenf[0 * npix + p] == Catch::Approx(expectedGreenMonth0[p]).margin(1e-4));
            CHECK(greenf[6 * npix + p] == Catch::Approx(expectedGreenMonth6[p]).margin(1e-4));
        }

        // LANDUSEF: nonzero (1-indexed category, fraction) pairs per pixel,
        // in row-major (south_north, west_east) order - deliberately
        // includes w2w's own real behavior of the per-pixel sum landing
        // below 1.0 at a handful of pixels (e.g. (2,2)/(2,3) end up with NO
        // nonzero category at all), not something to "fix" in the port.
        const std::vector<std::vector<std::pair<int, float>>> expectedLuf{
            {{7, 0.5f}, {12, 0.5f}}, {{12, 1.0f}}, {{7, 1.0f}}, {{7, 1.0f}}, {{14, 1.0f}},
            {{12, 0.5f}, {14, 0.5f}}, {{12, 1.0f}}, {{12, 1.0f}}, {{12, 1.0f}}, {{12, 1.0f}},
            {{12, 1.0f}}, {{12, 1.0f}}, {}, {}, {{12, 1.0f}},
            {{12, 1.0f}}, {{5, 1.0f}}, {{12, 1.0f}}, {{12, 1.0f}}, {{12, 1.0f}},
            {{12, 0.75f}}, {{14, 0.5f}}, {{12, 0.5f}, {14, 0.5f}}, {{12, 0.5f}, {14, 0.5f}}, {{14, 1.0f}},
        };
        const auto luf = result.readFloat("LANDUSEF");
        REQUIRE(result.shape("LANDUSEF") == std::vector<std::size_t>{1, 41, ny, nx});
        for (std::size_t p = 0; p < npix; ++p) {
            for (std::size_t cat = 0; cat < 41; ++cat) {
                const auto match = std::find_if(expectedLuf[p].begin(), expectedLuf[p].end(), [&](const auto& kv) { return kv.first == static_cast<int>(cat) + 1; });
                const float expected = match != expectedLuf[p].end() ? match->second : 0.0f;
                CAPTURE(p, cat);
                CHECK(luf[cat * npix + p] == Catch::Approx(expected).margin(1e-4));
            }
        }
    }

    std::filesystem::remove(dst);
    std::filesystem::remove(out);
}

TEST_CASE("removeUrban's add_wrf_version pre-pass collapses an already-LCZ-tagged pixel before the normal removal loop") {
    // No shipped fixture already carries LCZ-range LU_INDEX values (the
    // point of the pre-pass is re-running against w2w's OWN prior output,
    // which none of these fixtures are), so hand-mutate one pixel of a
    // real 41-category file into what a prior w2w run would have left
    // behind: category 35 (an LCZ class, LU_INDEX = 31 + ADD_LCZ_INT-30 =
    // 35 for a 41-category file) instead of any real land-use category.
    const auto dst = lczFixtureCopy("remove_urban_prepass");
    {
        auto file = NetcdfFile::open(dst, NetcdfFile::Mode::ReadWrite);
        auto lu = file.readFloat("LU_INDEX");
        lu[0] = 35.0f;  // pixel (0,0)
        file.writeFloat("LU_INDEX", lu);
        auto luf = file.readFloat("LANDUSEF");
        constexpr std::size_t npix = 25;
        luf[34 * npix + 0] = 1.0f;  // category 35 (index 34), pixel 0
        file.writeFloat("LANDUSEF", luf);
    }

    const auto out = std::filesystem::temp_directory_path() / "wrftools-cpp-netcdf-file-test-remove-urban-prepass-out.nc";
    removeUrban(dst, out, 3, 9);

    // Scoped so `result` closes before removing `out` below - an open
    // handle onto it blocks std::filesystem::remove on Windows.
    {
        const auto result = NetcdfFile::open(out, NetcdfFile::Mode::ReadOnly);
        const auto lu = result.readFloat("LU_INDEX");
        const auto luf = result.readFloat("LANDUSEF");
        constexpr std::size_t npix = 25;
        CHECK(lu[0] != 35.0f);   // no longer an LCZ class...
        CHECK(lu[0] != 13.0f);   // ...nor still urban (ISURBAN) - the normal removal loop ran on it
        CHECK(luf[34 * npix + 0] == 0.0f);  // category 35's fraction was collapsed into ISURBAN, then...
        CHECK(luf[12 * npix + 0] == 0.0f);  // ...ISURBAN's (index 12) fraction was itself cleared by the normal removal loop
    }

    std::filesystem::remove(dst);
    std::filesystem::remove(out);
}

TEST_CASE("removeUrban rejects an NPIX_AREA larger than the domain") {
    const auto dst = lczFixtureCopy("remove_urban_area_too_big");
    const auto out = std::filesystem::temp_directory_path() / "wrftools-cpp-netcdf-file-test-remove-urban-toobig-out.nc";
    CHECK_THROWS_AS(removeUrban(dst, out, 3, 1000), UserError);
    std::filesystem::remove(dst);
}

TEST_CASE("loadUcpTable parses w2w's own default lookup table") {
    const auto table = loadUcpTable("resources/lcz_ucp_lookup.csv");
    REQUIRE(table.size() == 17);
    const auto& lcz1 = table.at(1);
    CHECK(lcz1.frcUrb2d == Catch::Approx(0.95));
    CHECK(lcz1.mhUrb2dMin == Catch::Approx(25));
    CHECK(lcz1.mhUrb2d == Catch::Approx(50));
    CHECK(lcz1.mhUrb2dMax == Catch::Approx(75));
    CHECK(lcz1.bldfrUrb2d == Catch::Approx(0.5));
    CHECK(lcz1.h2w == Catch::Approx(2.5));
    const auto& lcz17 = table.at(17);
    CHECK(lcz17.frcUrb2d == Catch::Approx(0.0));
    checkCustomUcpTableIntegrity(table);  // shouldn't throw
}

TEST_CASE("defaultUcpTable's hardwired content matches the bundled resources/lcz_ucp_lookup.csv file byte-for-byte") {
    // defaultUcpTable() embeds its own copy of this CSV as a string literal
    // (see kDefaultUcpTableCsv in lcz.cpp) so LczForm never depends on
    // resolving a runtime file path - this pins the two against each
    // other so the embedded copy can't silently drift from the file a
    // packager or a --lcz-ucp-curious developer would actually look at.
    const auto fromFile = loadUcpTable("resources/lcz_ucp_lookup.csv");
    const auto& hardwired = defaultUcpTable();
    REQUIRE(hardwired.size() == fromFile.size());
    for (const auto& [lczClass, row] : fromFile) {
        REQUIRE(hardwired.contains(lczClass));
        const auto& h = hardwired.at(lczClass);
        CHECK(h.frcUrb2d == Catch::Approx(row.frcUrb2d));
        CHECK(h.mhUrb2dMin == Catch::Approx(row.mhUrb2dMin));
        CHECK(h.mhUrb2d == Catch::Approx(row.mhUrb2d));
        CHECK(h.mhUrb2dMax == Catch::Approx(row.mhUrb2dMax));
        CHECK(h.bldfrUrb2d == Catch::Approx(row.bldfrUrb2d));
        CHECK(h.h2w == Catch::Approx(row.h2w));
    }
}

TEST_CASE("loadUcpTableFromString parses the same content as loadUcpTable from a file") {
    const std::ifstream fileStream("resources/lcz_ucp_lookup.csv");
    std::ostringstream contents;
    contents << fileStream.rdbuf();
    const auto table = loadUcpTableFromString(contents.str());
    REQUIRE(table.size() == 17);
    CHECK(table.at(1).frcUrb2d == Catch::Approx(0.95));
}

TEST_CASE("loadUcpTable tolerates whitespace in headers and values") {
    const auto table = loadUcpTable("tests/fixtures/lcz/custom_lcz_ucp_w_spaces.csv");
    REQUIRE(table.size() == 17);
    CHECK(table.at(1).frcUrb2d == Catch::Approx(0.85));
    CHECK(table.at(4).mhUrb2d == Catch::Approx(50));
}

TEST_CASE("checkCustomUcpTableIntegrity rejects MH_URB2D_MIN >= MH_URB2D") {
    auto table = loadUcpTable("resources/lcz_ucp_lookup.csv");
    table.at(3).mhUrb2dMin = table.at(3).mhUrb2d + 1.0;  // now violates MIN < MH
    CHECK_THROWS_AS(checkCustomUcpTableIntegrity(table), UserError);
}

TEST_CASE("checkLczIntegrity passes an already-WGS84 LCZ GeoTIFF through unchanged and confirms it covers the WRF domain") {
    const auto wrf = NetcdfFile::open("tests/fixtures/lcz/geo_em.d04.nc", NetcdfFile::Mode::ReadOnly);
    const auto clean = checkLczIntegrity("tests/fixtures/lcz/lcz_zaragoza.tif", 0, wrf);

    CHECK(clean.width == 1210);
    CHECK(clean.height == 765);
    // Pinned against a live check_lcz_integrity() run (add_wrf_version,
    // 7801b3e) on this exact fixture pair.
    CHECK(clean.geotransform[0] == Catch::Approx(-1.7002029439153534));
    CHECK(clean.geotransform[1] == Catch::Approx(0.0013474837091883177));
    CHECK(clean.geotransform[2] == Catch::Approx(0.0));
    CHECK(clean.geotransform[3] == Catch::Approx(42.195661701819475));
    CHECK(clean.geotransform[4] == Catch::Approx(0.0));
    CHECK(clean.geotransform[5] == Catch::Approx(-0.0013474837091883162));
    REQUIRE(clean.values.size() == static_cast<std::size_t>(clean.width) * clean.height);
    CHECK(clean.values[0] == 14.0f);  // row 0, col 0
    CHECK(clean.values[static_cast<std::size_t>(clean.height / 2) * clean.width + clean.width / 2] == 8.0f);  // row 382, col 605
}

TEST_CASE("checkLczIntegrity reprojects a UTM LCZ GeoTIFF and relabels 100-series class codes") {
    const auto wrf = NetcdfFile::open("tests/fixtures/lcz/geo_em.d02_Shanghai.nc", NetcdfFile::Mode::ReadOnly);
    const auto clean = checkLczIntegrity("tests/fixtures/lcz/Shanghai.tif", 0, wrf);

    // Pinned against a live check_lcz_integrity() run (add_wrf_version,
    // 7801b3e): testing/Shanghai.tif is UTM zone 51N with 100-series
    // (LCZ Generator) class codes and a real -1 NoData value - this one
    // fixture exercises the reprojection, relabeling, AND NoData-exclusion
    // paths all at once.
    CHECK(clean.width == 1001);
    CHECK(clean.height == 904);
    CHECK(clean.geotransform[0] == Catch::Approx(121.06546942474998));
    CHECK(clean.geotransform[1] == Catch::Approx(0.0009768493252783942));
    CHECK(clean.geotransform[3] == Catch::Approx(31.662660675894173));
    CHECK(clean.geotransform[5] == Catch::Approx(-0.0009768493252784018));

    const auto at = [&](int row, int col) { return clean.values[static_cast<std::size_t>(row) * clean.width + col]; };
    CHECK(at(0, 0) == 0.0f);      // outside the source UTM tile's coverage -> NoData -> clamped to 0
    CHECK(at(903, 1000) == 0.0f);
    CHECK(at(452, 500) == 14.0f);  // center
    CHECK(at(10, 10) == 0.0f);
    CHECK(at(100, 100) == 14.0f);
    CHECK(at(500, 500) == 4.0f);
    CHECK(at(900, 50) == 14.0f);
    // Every value is either 0 (NoData) or a valid 1-17 LCZ class - the
    // 100-series relabeling ran and covered the whole raster (a partial
    // relabel would leave stray 101-107 values behind).
    for (float v : clean.values) CHECK((v == 0.0f || (v >= 1.0f && v <= 17.0f)));
}

TEST_CASE("checkLczIntegrity rejects an LCZ GeoTIFF that doesn't cover the WRF domain") {
    const auto wrf = NetcdfFile::open("tests/fixtures/lcz/geo_em.d04.nc", NetcdfFile::Mode::ReadOnly);
    CHECK_THROWS_AS(checkLczIntegrity("tests/fixtures/lcz/lcz_too_small.tif", 0, wrf), UserError);
}

TEST_CASE("Full LCZ pipeline (checkLczIntegrity -> removeUrban -> createLczParamsFile -> createLczExtentFile) "
    "matches a live add_wrf_version run on the real Zaragoza sample domain") {
    const auto origPath = std::filesystem::path("tests/fixtures/lcz/geo_em.d04.nc");
    const auto noUrbanPath = std::filesystem::temp_directory_path() / "wrftools-cpp-lcz-e2e-NoUrban.nc";
    const auto paramsPath = std::filesystem::temp_directory_path() / "wrftools-cpp-lcz-e2e-LCZ_params.nc";
    const auto extentPath = std::filesystem::temp_directory_path() / "wrftools-cpp-lcz-e2e-LCZ_extent.nc";
    for (const auto& p : {noUrbanPath, paramsPath, extentPath}) std::filesystem::remove(p);

    const auto wrf = NetcdfFile::open(origPath, NetcdfFile::Mode::ReadOnly);
    const auto clean = checkLczIntegrity("tests/fixtures/lcz/lcz_zaragoza.tif", 0, wrf);
    removeUrban(origPath, noUrbanPath, 45, std::nullopt);

    const std::vector<int> builtLcz{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const auto ucpTable = loadUcpTable("resources/lcz_ucp_lookup.csv");
    LczParamsInputs inputs{noUrbanPath, origPath, clean, builtLcz, ucpTable, WrfVersionInfo{30, 41}, 0.2};
    const int nbuiMax = createLczParamsFile(inputs, paramsPath);

    // Pinned against a live add_wrf_version (7801b3e) run of the full
    // pipeline (check_lcz_integrity -> wrf_remove_urban ->
    // create_lcz_params_file -> create_lcz_extent_file) against this
    // exact fixture pair (w2w's own sample_data/), NPIX_NLC=45,
    // FRC_THRESHOLD=0.2, BUILT_LCZ=1..10. nbui_max=5 also matches the
    // README's own documented note for this sample dataset.
    CHECK(nbuiMax == 5);

    // Scoped so `params` closes before this test removes paramsPath at
    // the end - an open handle onto it blocks std::filesystem::remove on
    // Windows.
    {
        const auto params = NetcdfFile::open(paramsPath, NetcdfFile::Mode::ReadOnly);
        CHECK(params.getAttribute("", "NUM_LAND_CAT").numbers[0] == 41);
        CHECK(params.getAttribute("", "FLAG_URB_PARAM").numbers[0] == 1);
        CHECK(params.getAttribute("", "NBUI_MAX").numbers[0] == 5);

        // metgrid.exe reads FieldType/MemoryOrder/stagger off every geo_em
        // field and aborts if any is missing - both rebuilt variables must
        // carry them.
        for (const std::string var : {"FRC_URB2D", "URB_PARAM"}) {
            INFO(var);
            REQUIRE(params.hasAttribute(var, "FieldType"));
            CHECK(params.getAttribute(var, "FieldType").numbers[0] == 104);
            REQUIRE(params.hasAttribute(var, "MemoryOrder"));
            CHECK(params.getAttribute(var, "MemoryOrder").text == (var == "URB_PARAM" ? "XYZ" : "XY"));
            REQUIRE(params.hasAttribute(var, "stagger"));
            CHECK(params.getAttribute(var, "stagger").text == "M");
            CHECK(params.hasAttribute(var, "sr_x"));
            CHECK(params.hasAttribute(var, "sr_y"));
        }

        constexpr std::size_t ny = 102, nx = 162, npix = ny * nx;
        const auto luIndex = params.readFloat("LU_INDEX");
        const auto frcUrb2d = params.readFloat("FRC_URB2D");
        const auto urbParam = params.readFloat("URB_PARAM");
        const auto landusef = params.readFloat("LANDUSEF");
        const auto greenfrac = params.readFloat("GREENFRAC");
        REQUIRE(luIndex.size() == npix);
        REQUIRE(urbParam.size() == 132 * npix);
        REQUIRE(landusef.size() == 41 * npix);

        // A specific pixel (50, 80) with a real LCZ class assignment, checked
        // to Python's float32 precision.
        const std::size_t p = 50 * nx + 80;
        CHECK(luIndex[p] == Catch::Approx(38.0f));
        CHECK(frcUrb2d[p] == Catch::Approx(0.80955416f));
        CHECK(urbParam[90 * npix + p] == Catch::Approx(0.48248515f));   // LP_URB2D
        CHECK(urbParam[91 * npix + p] == Catch::Approx(12.082176f));    // MH_URB2D
        CHECK(urbParam[92 * npix + p] == Catch::Approx(2.7649412f));    // STDH_URB2D
        CHECK(urbParam[93 * npix + p] == Catch::Approx(11.26162f));     // HGT_URB2D
        CHECK(urbParam[94 * npix + p] == Catch::Approx(0.9502786f));    // LB_URB2D
        for (int i = 0; i < 4; ++i) CHECK(urbParam[static_cast<std::size_t>(95 + i) * npix + p] == Catch::Approx(0.46779343f));  // LF_URB2D x4
        const std::vector<float> expectedHiBins{8.883777618408203f, 40.369163513183594f, 12.222759246826172f, 26.285173416137695f,
            12.239124298095703f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        // hiResample computes these analytically (see its doc comment); w2w.py
        // draws 100000 Monte Carlo samples from the same truncated-normal
        // distribution and bins them, so its own reference values here carry
        // real sampling noise (binomial std error on the order of ~0.1-0.2
        // percentage points for bins in this range) - a wide-ish margin is
        // the correct comparison, not a sign of imprecision on this port's
        // side.
        for (int b = 0; b < 15; ++b)
            CHECK(urbParam[static_cast<std::size_t>(117 + b) * npix + p] == Catch::Approx(expectedHiBins[static_cast<std::size_t>(b)]).margin(0.1));
        CHECK(landusef[37 * npix + p] == Catch::Approx(1.0f));
        CHECK(greenfrac[0 * npix + p] == Catch::Approx(0.29356405f));
        CHECK(greenfrac[6 * npix + p] == Catch::Approx(0.3034502f));

        // Aggregate checks across the whole domain.
        CHECK(*std::max_element(frcUrb2d.begin(), frcUrb2d.end()) == Catch::Approx(0.9f));
        for (std::size_t q = 0; q < npix; ++q) {
            double sum = 0.0;
            for (std::size_t cat = 0; cat < 41; ++cat) sum += landusef[cat * npix + q];
            CAPTURE(q);
            CHECK(sum <= Catch::Approx(1.0).margin(1e-4));
        }
        // NaN LU_INDEX values are a REAL w2w.py behavior in principle (an
        // LCZ-mode-resample coverage gap at an otherwise FRC_URB2D>0 pixel
        // writes straight through into LU_INDEX, unguarded) - w2w.py's own
        // live run against this exact fixture produces 205 such pixels.
        // Checked at those 205 exact positions here (not by raw count): this
        // port's direct GDALReprojectImage call resolves a real class at
        // EVERY one of them instead (confirmed by probing lczResample
        // directly) - i.e. strictly more complete, never NaN where Python
        // wasn't already, and never a bogus class elsewhere. That's most
        // likely a default-option difference between GDALReprojectImage and
        // rasterio's own reproject() wrapper for GRA_Mode's edge-of-coverage
        // handling specifically, not a masking/algorithm bug - every other
        // value in this pixel-by-pixel and aggregate comparison (33000+
        // assertions) matches Python to float32 precision. A real NaN, if
        // this port's warp ever does produce one (a still-supported case -
        // see createLczParamsFile's own doc comment on where it can come
        // from), must never silently become a bogus finite class number.
        for (float v : luIndex) CHECK((std::isnan(v) || (v >= 1.0f && v <= 60.0f)));
    }

    createLczExtentFile(paramsPath, origPath, extentPath);
    // Scoped for the same reason as `params` above - `extent` must close
    // before this test removes extentPath at the end.
    {
        const auto extent = NetcdfFile::open(extentPath, NetcdfFile::Mode::ReadOnly);
        CHECK(extent.getAttribute("", "NUM_LAND_CAT").numbers[0] == 41);
        CHECK(extent.getAttribute("", "FLAG_URB_PARAM").numbers[0] == 0);
        const auto extentLu = extent.readFloat("LU_INDEX");
        // Every non-NaN LU_INDEX value in the extent file is either a real
        // (non-LCZ) category or plain ISURBAN (13) - the LCZ classes (31-40)
        // were all collapsed back.
        for (float v : extentLu) {
            if (std::isnan(v)) continue;
            const int cls = static_cast<int>(std::lround(v));
            CAPTURE(cls);
            CHECK_FALSE((cls >= 31 && cls <= 40));
        }
        CHECK(std::count(extentLu.begin(), extentLu.end(), 13.0f) > 0);
    }

    for (const auto& fp : {noUrbanPath, paramsPath, extentPath}) std::filesystem::remove(fp);
}

TEST_CASE("NetcdfFile::rebuildStructure resizes a source dimension that already collides with an override's new dimension name") {
    // Real-world regression: a geo_em file straight out of a newer
    // geogrid.exe run can already have urban physics configured, with its
    // own "num_urb_params" dimension - the EXACT name createLczParamsFile
    // needs for its own URB_PARAM override. Before this fix,
    // rebuildStructure always ran an unconditional nc_def_dim for every
    // `newDimensions` entry, which failed with "String match to name in
    // use" the instant a source file already had that name (confirmed
    // against a real user-supplied Hong Kong domain: geogrid.exe's own
    // output already carries `num_urb_params = 132`). 5by5.nc doesn't
    // have this dimension itself (its own URB_PARAM uses an oddly-named
    // "18" dimension, like the Zaragoza sample data) - added here to
    // reproduce the collision deliberately.
    const auto dst = lczFixtureCopy("rebuild_collision");
    {
        auto file = NetcdfFile::open(dst, NetcdfFile::Mode::ReadWrite);
        file.defineDimension("num_urb_params", 18);
        file.defineVariable("OTHER_URB_VAR", NC_FLOAT, {"Time", "num_urb_params", "south_north", "west_east"});
        std::vector<float> data(1 * 18 * 5 * 5, 7.0f);
        file.writeFloat("OTHER_URB_VAR", data);
    }

    const std::size_t southNorth = 5, westEast = 5, newLandCat = 45, newNumUrbParams = 132;
    const std::vector<float> urbParamData(1 * newNumUrbParams * southNorth * westEast, 3.0f);
    NetcdfFile::rebuildStructure(
        dst, "land_cat", newLandCat,
        [&](const std::string& variableName) -> std::optional<std::vector<float>> {
            if (variableName != "LANDUSEF") return std::nullopt;
            return std::vector<float>(newLandCat * southNorth * westEast, 0.0f);
        },
        {{"num_urb_params", newNumUrbParams}},  // the caller still declares it as "new" - it happens to already exist in this source file
        {{"URB_PARAM", NC_FLOAT, {"Time", "num_urb_params", "south_north", "west_east"}, urbParamData}});

    // Scoped so `reopened` closes before removing `dst` below - an open
    // handle onto it blocks std::filesystem::remove on Windows.
    {
        const auto reopened = NetcdfFile::open(dst, NetcdfFile::Mode::ReadOnly);
        const auto dims = reopened.dimensions();
        const auto numUrbParamsDim = std::find_if(dims.begin(), dims.end(), [](const auto& d) { return d.name == "num_urb_params"; });
        REQUIRE(numUrbParamsDim != dims.end());
        CHECK(numUrbParamsDim->length == newNumUrbParams);  // resized in place, not left at 18

        CHECK(reopened.shape("URB_PARAM") == std::vector<std::size_t>{1, newNumUrbParams, southNorth, westEast});
        const auto urbParam = reopened.readFloat("URB_PARAM");
        CHECK(urbParam.size() == newNumUrbParams * southNorth * westEast);
        for (float v : urbParam) CHECK(v == 3.0f);

        // OTHER_URB_VAR wasn't overridden and still uses the (now-resized)
        // "num_urb_params" dimension - its original 18 slots' worth of data
        // must survive unchanged; slots 18..131 are new, netCDF-default-filled
        // space this test doesn't assert on.
        CHECK(reopened.shape("OTHER_URB_VAR") == std::vector<std::size_t>{1, newNumUrbParams, southNorth, westEast});
        const auto otherUrbVar = reopened.readFloat("OTHER_URB_VAR");
        for (std::size_t p = 0; p < 18 * southNorth * westEast; ++p) CHECK(otherUrbVar[p] == 7.0f);
    }

    std::filesystem::remove(dst);
}

TEST_CASE("NetcdfFile::resizeDimension grows LANDUSEF's land_cat dimension, matching w2w's own category-count expansion") {
    const auto dst = lczFixtureCopy("resize");

    // This fixture is already a 41-category file; resize to 45 (rather
    // than a same-size no-op) to genuinely exercise the dimension-growth
    // path w2w's own 21->41/41->61 expansions rely on.
    const std::size_t southNorth = 5, westEast = 5, newLandCat = 45;
    NetcdfFile::resizeDimension(dst, "land_cat", newLandCat, [&](const std::string& variableName) -> std::optional<std::vector<float>> {
        if (variableName != "LANDUSEF") return std::nullopt;
        std::vector<float> data(newLandCat * southNorth * westEast, 0.0f);
        // Category 1 (index 0) covers every pixel - mirrors
        // w2w._adjust_greenfrac_landusef's zero-then-set-one pattern.
        for (std::size_t i = 0; i < southNorth * westEast; ++i) data[i] = 1.0f;
        return data;
    });

    // Scoped so `reopened` closes before removing `dst` below - an open
    // handle onto it blocks std::filesystem::remove on Windows.
    {
        const auto reopened = NetcdfFile::open(dst, NetcdfFile::Mode::ReadOnly);
        const auto dims = reopened.dimensions();
        const auto landCat = std::find_if(dims.begin(), dims.end(), [](const auto& d) { return d.name == "land_cat"; });
        REQUIRE(landCat != dims.end());
        CHECK(landCat->length == newLandCat);

        CHECK(reopened.shape("LANDUSEF") == std::vector<std::size_t>{1, newLandCat, southNorth, westEast});
        const auto landusef = reopened.readFloat("LANDUSEF");
        CHECK(landusef.size() == newLandCat * southNorth * westEast);
        for (std::size_t i = 0; i < southNorth * westEast; ++i) CHECK(landusef[i] == 1.0f);
        for (std::size_t i = southNorth * westEast; i < landusef.size(); ++i) CHECK(landusef[i] == 0.0f);

        // A variable untouched by the resize (different dimensions entirely)
        // still round-trips correctly.
        CHECK(reopened.shape("LU_INDEX") == std::vector<std::size_t>{1, southNorth, westEast});
        CHECK(reopened.getAttribute("", "MMINLU").text == "MODIFIED_IGBP_MODIS_NOAH");
    }

    std::filesystem::remove(dst);
}

namespace {
// Lays out a fresh temp directory with canonically-named geo_em.dNN.nc
// files copied from the given fixture paths (index-1 = filenames[0], etc.)
// - expandLandCatParents locates parent domains purely by filename pattern
// next to `dstFile`, matching w2w.py's own Info.dst_file-relative lookup.
std::filesystem::path expandLandCatParentsFixture(const std::string& suffix, const std::vector<std::pair<int, std::string>>& domainToFixture) {
    const auto dir = std::filesystem::temp_directory_path() / ("wrftools-cpp-expand-land-cat-parents-" + suffix);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    for (const auto& [domain, fixture] : domainToFixture) {
        const auto dst = dir / std::format("geo_em.d{:02d}.nc", domain);
        std::filesystem::copy_file(std::filesystem::path("tests/fixtures/lcz") / fixture, dst);
    }
    return dir;
}
}  // namespace

TEST_CASE("expandLandCatParents rejects a filename with no two-digit domain number rather than crashing") {
    // A raw std::stoi on a non-numeric substring throws std::invalid_argument,
    // not UserError - this must be caught and converted before it ever
    // reaches a caller (Stage 5's LczForm lets a user pick an arbitrary
    // target file), see the isdigit guard in lcz.cpp.
    CHECK_THROWS_AS(expandLandCatParents("build/not_a_geo_em_file.nc", WrfVersionInfo{30, 41}), UserError);
    CHECK_THROWS_AS(expandLandCatParents("build/geo_em.dXX.nc", WrfVersionInfo{30, 41}), UserError);
}

TEST_CASE("expandLandCatParents warns for every missing parent domain file") {
    const auto dir = expandLandCatParentsFixture("missing", {{4, "geo_em.d04.nc"}});
    const auto messages = expandLandCatParents(dir / "geo_em.d04.nc", WrfVersionInfo{30, 41});

    REQUIRE(messages.size() == 3);
    for (const auto& m : messages) CHECK(m.find("not found") != std::string::npos);
}

TEST_CASE("expandLandCatParents reports an unreadable parent domain rather than throwing") {
    const auto dir = expandLandCatParentsFixture(
        "unreadable", {{1, "geo_em.d01_Shanghai_no_NUM_LAND_CAT.nc"}, {2, "geo_em.d02_Shanghai.nc"}});
    const auto messages = expandLandCatParents(dir / "geo_em.d02.nc", WrfVersionInfo{30, 41});

    REQUIRE(messages.size() == 1);
    CHECK(messages[0].find("Cannot read NUM_LAND_CAT") != std::string::npos);
}

TEST_CASE("expandLandCatParents leaves an already-correct parent domain untouched") {
    const auto dir = expandLandCatParentsFixture("already41", {{1, "geo_em.d01_Shanghai_ncl41.nc"}, {2, "geo_em.d02_Shanghai.nc"}});
    const auto messages = expandLandCatParents(dir / "geo_em.d02.nc", WrfVersionInfo{30, 41});

    REQUIRE(messages.size() == 1);
    CHECK(messages[0].find("already contains 41 LC classes") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(dir / "geo_em.d01_41.nc"));
}

TEST_CASE("expandLandCatParents grows a mismatched parent domain's LANDUSEF and writes a suffixed file") {
    const auto dir = expandLandCatParentsFixture("mismatch", {{1, "geo_em.d01_Shanghai_ncl20.nc"}, {2, "geo_em.d02_Shanghai.nc"}});
    const auto messages = expandLandCatParents(dir / "geo_em.d02.nc", WrfVersionInfo{30, 41});

    REQUIRE(messages.size() == 1);
    CHECK(messages[0].find("expanded to 41 LC classes") != std::string::npos);

    const auto ofile = dir / "geo_em.d01_41.nc";
    REQUIRE(std::filesystem::exists(ofile));
    const auto grown = NetcdfFile::open(ofile, NetcdfFile::Mode::ReadOnly);
    CHECK(grown.getAttribute("", "NUM_LAND_CAT").numbers[0] == 41);
    const auto shape = grown.shape("LANDUSEF");
    CHECK(shape[1] == 41);
    // The original file's own 20 categories are preserved verbatim; the
    // newly-added ones are zero-filled.
    const auto original = NetcdfFile::open(dir / "geo_em.d01.nc", NetcdfFile::Mode::ReadOnly);
    const auto origLanduseF = original.readFloat("LANDUSEF");
    const auto grownLanduseF = grown.readFloat("LANDUSEF");
    const std::size_t npix = shape[2] * shape[3];
    for (std::size_t p = 0; p < npix; ++p) CHECK(grownLanduseF[p] == origLanduseF[p]);
    for (std::size_t cat = 20; cat < 41; ++cat)
        for (std::size_t p = 0; p < npix; ++p) CHECK(grownLanduseF[cat * npix + p] == 0.0f);
    CHECK(grown.getAttribute("LANDUSEF", "description").text == "Noah-modified 41-category IGBP-MODIS landuse");
}

namespace {
std::filesystem::path checksAndCleaningFixtureCopy(const std::string& fixture, const std::string& suffix) {
    const auto dst = std::filesystem::temp_directory_path() / ("wrftools-cpp-checks-and-cleaning-" + suffix + "-" + fixture);
    std::filesystem::copy_file(std::filesystem::path("tests/fixtures/lcz") / fixture, dst, std::filesystem::copy_options::overwrite_existing);
    return dst;
}
}  // namespace

TEST_CASE("checksAndCleaning passes every check against the real add_wrf_version pipeline output and deletes the clean LCZ tif") {
    const auto srcClean = checksAndCleaningFixtureCopy("lcz_zaragoza.tif", "ok");  // stand-in for the *_clean.tif this function deletes
    REQUIRE(std::filesystem::exists(srcClean));

    const ChecksAndCleaningInputs inputs{
        srcClean,
        "tests/fixtures/lcz/geo_em.d04.nc",
        "tests/fixtures/lcz/geo_em.d04_NoUrban.nc",
        "tests/fixtures/lcz/geo_em.d04_LCZ_extent.nc",
        "tests/fixtures/lcz/geo_em.d04_LCZ_params.nc",
    };
    const auto ucpTable = loadUcpTable("resources/lcz_ucp_lookup.csv");
    const auto results = checksAndCleaning(inputs, ucpTable);

    REQUIRE(results.size() == 9);
    for (const auto& r : results) CHECK(r.status == CheckStatus::Ok);
    CHECK_FALSE(std::filesystem::exists(srcClean));
}

TEST_CASE("checksAndCleaning warns on checks 1-5 against dummy pipeline output missing the LCZ params content") {
    const ChecksAndCleaningInputs inputs{
        "tests/fixtures/lcz/does_not_exist_clean.tif",  // nonexistent - cleanup step is a no-op
        "tests/fixtures/lcz/geo_em.d04.nc",
        "tests/fixtures/lcz/geo_em.d04_NoUrban_dummy.nc",
        "tests/fixtures/lcz/geo_em.d04_LCZ_extent_dummy.nc",
        "tests/fixtures/lcz/geo_em.d04_LCZ_params_dummy.nc",
    };
    const auto ucpTable = loadUcpTable("resources/lcz_ucp_lookup.csv");
    const auto results = checksAndCleaning(inputs, ucpTable);

    // The dummy params file carries neither FRC_URB2D nor URB_PARAM, so
    // checks 6/7/8 are skipped entirely (gated the same way w2w.py's own
    // checks_and_cleaning gates them) - only checks 1, 2, 3, 4, 5, 9 run.
    REQUIRE(results.size() == 6);
    for (std::size_t i = 0; i < 5; ++i) {
        INFO(results[i].name);
        CHECK(results[i].status == CheckStatus::Warning);
    }
}

TEST_CASE("checksAndCleaning warns on checks 6-9 against a dummy LCZ params file with out-of-range URB_PARAM values") {
    const ChecksAndCleaningInputs inputs{
        "tests/fixtures/lcz/does_not_exist_clean.tif",
        "tests/fixtures/lcz/geo_em.d04.nc",
        "tests/fixtures/lcz/geo_em.d04_NoUrban_dummy.nc",
        "tests/fixtures/lcz/geo_em.d04_LCZ_extent_dummy.nc",
        "tests/fixtures/lcz/geo_em.d04_LCZ_params_with_ucp_dummy.nc",
    };
    const auto ucpTable = loadUcpTable("resources/lcz_ucp_lookup.csv");
    const auto results = checksAndCleaning(inputs, ucpTable);

    REQUIRE(results.size() == 9);
    const auto find = [&](const std::string& prefix) -> const CheckResult& {
        for (const auto& r : results)
            if (r.name.starts_with(prefix)) return r;
        FAIL("no check result named " << prefix);
        throw std::logic_error("unreachable");
    };
    CHECK(find("Check 6").status == CheckStatus::Warning);
    CHECK(find("Check 7").status == CheckStatus::Warning);
    CHECK(find("Check 8").status == CheckStatus::Warning);
    CHECK(find("Check 9").status == CheckStatus::Warning);
}

TEST_CASE("checksAndCleaning rejects an unsupported original NUM_LAND_CAT") {
    const auto dst = lczFixtureCopy("checks_unsupported_num_land_cat");
    {
        auto file = NetcdfFile::open(dst, NetcdfFile::Mode::ReadWrite);
        NetcdfFile::Attribute a;
        a.name = "NUM_LAND_CAT";
        a.type = NC_INT;
        a.numbers = {24.0};
        file.putAttribute("", a);
    }
    const ChecksAndCleaningInputs inputs{"tests/fixtures/lcz/does_not_exist_clean.tif", dst, dst, dst, dst};
    const auto ucpTable = loadUcpTable("resources/lcz_ucp_lookup.csv");
    CHECK_THROWS_AS(checksAndCleaning(inputs, ucpTable), UserError);
    std::filesystem::remove(dst);
}

// ---- Reproject tab (see PORT.md) ----
//
// Fixtures under tests/fixtures/reproject/ are carved (via `ncks`, not
// synthesized) from a real production run - three consecutive half-hourly
// files from /mnt/DC550/HONGKONG/results/PERIOD16_new, a WRF 4.7.1 Mercator
// domain over Hong Kong (250 m DX/DY) - trimmed to a handful of variables
// and vertical levels to keep them small while keeping real geometry,
// attributes, and projection. See the variable list below for why these
// four were kept: T2 (plain 2D), LU_INDEX (categorical 2D), U (X-staggered
// 3D on bottom_top), W (Z-staggered 3D on bottom_top_stag - the
// destaggering case), TSLB (3D on soil_layers_stag - the "stag" in the name
// that must NOT be destaggered).
namespace {
std::filesystem::path reprojectOutputDir(const std::string& suffix) {
    const auto dir = std::filesystem::temp_directory_path() / ("wrftools-cpp-reproject-" + suffix);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}
}  // namespace

TEST_CASE("describeTargetCrs resolves a geographic and a projected EPSG code, and rejects an unknown one") {
    const auto geographic = describeTargetCrs(4326);
    CHECK(geographic.isGeographic);
    CHECK(geographic.xName == "lon");
    CHECK(geographic.yName == "lat");
    CHECK(geographic.gridMappingName == "latitude_longitude");

    const auto projected = describeTargetCrs(32650);  // UTM 50N - Hong Kong's own zone
    CHECK_FALSE(projected.isGeographic);
    CHECK(projected.xName == "x");
    CHECK(projected.yName == "y");
    CHECK(projected.xUnits == "m");
    CHECK(projected.gridMappingName == "transverse_mercator");

    CHECK_THROWS_AS(describeTargetCrs(999999), UserError);
}

TEST_CASE("computeDestinationGrid honours pixel size and extent overrides, and rejects a degenerate request") {
    WrfFile file("tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc");
    const auto target = describeTargetCrs(4326);
    const auto suggested = computeDestinationGrid(file.size()[0], file.size()[1], file.projectionWkt(), file.geotransform(), target, {});
    CHECK(suggested.width > 0);
    CHECK(suggested.height > 0);

    GridOverride halved;
    halved.pixelSizeX = suggested.geotransform[1] * 2.0;
    halved.pixelSizeY = -suggested.geotransform[5] * 2.0;
    const auto coarser = computeDestinationGrid(file.size()[0], file.size()[1], file.projectionWkt(), file.geotransform(), target, halved);
    CHECK(coarser.width < suggested.width);
    CHECK(coarser.height < suggested.height);

    GridOverride degenerate;
    degenerate.pixelSizeX = 1e-12;
    degenerate.pixelSizeY = 1e-12;
    CHECK_THROWS_AS(computeDestinationGrid(file.size()[0], file.size()[1], file.projectionWkt(), file.geotransform(), target, degenerate), UserError);
}

TEST_CASE("runReproject writes a CF-1.7 file for a single real wrfout, with a real grid_mapping and no stale WRF projection globals") {
    const auto dir = reprojectOutputDir("single-file");
    ReprojectOptions options;
    options.inputs = {"tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc"};
    options.targetEpsg = 4326;
    options.variables = {"T2", "U"};
    options.outputDirectory = dir;
    const auto written = runReproject(options);
    REQUIRE(written.size() == 1);

    auto out = NetcdfFile::open(written.front(), NetcdfFile::Mode::ReadOnly);
    CHECK(out.getAttribute("", "Conventions").text == "CF-1.7");
    CHECK_FALSE(out.hasAttribute("", "MAP_PROJ"));
    CHECK(out.hasAttribute("", "WRF_MAP_PROJ"));
    CHECK(static_cast<int>(out.getAttribute("", "WRF_MAP_PROJ").numbers.at(0)) == 3);  // Mercator, unchanged from the source
    CHECK(out.getAttribute("", "MMINLU").text == "MODIFIED_IGBP_MODIS_NOAH");  // simulation metadata, carried forward verbatim

    const auto t2 = out.variable("T2");
    CHECK(t2.dimensionNames == std::vector<std::string>{"time", "lat", "lon"});
    CHECK(out.getAttribute("T2", "grid_mapping").text == "crs");
    CHECK(out.getAttribute("T2", "long_name").text == "TEMP at 2 M");
    CHECK(out.getAttribute("T2", "units").text == "K");

    const auto u = out.variable("U");
    CHECK(u.dimensionNames == std::vector<std::string>{"time", "bottom_top", "lat", "lon"});

    CHECK(out.getAttribute("crs", "grid_mapping_name").text == "latitude_longitude");
    CHECK_FALSE(out.getAttribute("crs", "crs_wkt").text.empty());

    // Bilinear resampling cannot invent values outside the source's own
    // finite range.
    WrfFile source("tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc");
    const auto sourceT2 = source.read("T2", 0, 0);
    float sourceMin = std::numeric_limits<float>::infinity(), sourceMax = -std::numeric_limits<float>::infinity();
    for (float v : sourceT2) if (std::isfinite(v)) { sourceMin = std::min(sourceMin, v); sourceMax = std::max(sourceMax, v); }
    const auto outT2 = out.readFloat("T2");
    bool sawFinite = false;
    for (float v : outT2) {
        if (v > 1e30f) continue;  // WRF fill value
        sawFinite = true;
        CHECK(v >= sourceMin - 0.5f);
        CHECK(v <= sourceMax + 0.5f);
    }
    CHECK(sawFinite);
}

TEST_CASE("runReproject vertically destaggers W onto bottom_top, but leaves soil_layers_stag alone") {
    const auto dir = reprojectOutputDir("vertical-destagger");
    ReprojectOptions options;
    options.inputs = {"tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc"};
    options.targetEpsg = 4326;
    options.variables = {"W", "TSLB"};
    options.outputDirectory = dir;
    const auto written = runReproject(options);
    auto out = NetcdfFile::open(written.front(), NetcdfFile::Mode::ReadOnly);

    // Source W has bottom_top_stag = 5 levels (see the fixture's own
    // ncdump); destaggered, it must land on the SAME "bottom_top" dimension
    // U/T/P already use, one level fewer.
    CHECK(out.variable("W").dimensionNames == std::vector<std::string>{"time", "bottom_top", "lat", "lon"});
    const auto bottomTop = out.dimensions();
    const auto found = std::find_if(bottomTop.begin(), bottomTop.end(), [](const auto& d) { return d.name == "bottom_top"; });
    REQUIRE(found != bottomTop.end());
    CHECK(found->length == 4);  // bottom_top_stag(5) - 1

    // soil_layers_stag keeps its own name and length unchanged - the
    // "_stag" in its name is WRF naming convention (soil-layer midpoint
    // depths), not a staggered grid.
    CHECK(out.variable("TSLB").dimensionNames == std::vector<std::string>{"time", "soil_layers_stag", "lat", "lon"});
    const auto soil = std::find_if(bottomTop.begin(), bottomTop.end(), [](const auto& d) { return d.name == "soil_layers_stag"; });
    REQUIRE(soil != bottomTop.end());
    CHECK(soil->length == 4);
}

TEST_CASE("runReproject forces nearest-neighbour for a categorical variable even when bilinear is requested") {
    const auto dir = reprojectOutputDir("categorical");
    WrfFile source("tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc");
    const auto sourceLu = source.read("LU_INDEX", 0, 0);
    std::set<int> sourceClasses;
    for (float v : sourceLu) if (std::isfinite(v)) sourceClasses.insert(static_cast<int>(std::lround(v)));

    ReprojectOptions options;
    options.inputs = {"tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc"};
    options.targetEpsg = 4326;
    options.variables = {"LU_INDEX"};
    options.resampling = ResampleMethod::Bilinear;
    options.nearestForCategorical = true;
    options.outputDirectory = dir;
    const auto written = runReproject(options);
    auto out = NetcdfFile::open(written.front(), NetcdfFile::Mode::ReadOnly);
    const auto outLu = out.readFloat("LU_INDEX");
    bool sawFinite = false;
    for (float v : outLu) {
        if (v > 1e30f) continue;
        sawFinite = true;
        CHECK(std::abs(v - std::lround(v)) < 1e-3f);  // integral - bilinear would have blended fractional values in
        CHECK(sourceClasses.contains(static_cast<int>(std::lround(v))));
    }
    CHECK(sawFinite);
}

TEST_CASE("runReproject merges a real 3-file series into one output with a correct CF time axis") {
    const auto dir = reprojectOutputDir("merge");
    ReprojectOptions options;
    options.inputs = {
        "tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc",
        "tests/fixtures/reproject/wrfout_d03_2025-03-14_00_30_00.nc",
        "tests/fixtures/reproject/wrfout_d03_2025-03-14_01_00_00.nc",
    };
    options.targetEpsg = 4326;
    options.variables = {"T2"};
    options.seriesMode = SeriesMode::Merge;
    options.outputDirectory = dir;
    const auto written = runReproject(options);
    REQUIRE(written.size() == 1);

    auto out = NetcdfFile::open(written.front(), NetcdfFile::Mode::ReadOnly);
    const auto dims = out.dimensions();
    const auto time = std::find_if(dims.begin(), dims.end(), [](const auto& d) { return d.name == "time"; });
    REQUIRE(time != dims.end());
    CHECK(time->length == 3);
    CHECK(out.getAttribute("time", "units").text == "seconds since 2025-03-14 00:00:00");
    const auto seconds = out.readDouble("time");
    REQUIRE(seconds.size() == 3);
    CHECK(seconds[0] == Catch::Approx(0.0));
    CHECK(seconds[1] == Catch::Approx(1800.0));
    CHECK(seconds[2] == Catch::Approx(3600.0));

    const auto timesShape = out.shape("Times");
    REQUIRE(timesShape.size() == 2);
    CHECK(timesShape[0] == 3);
    CHECK(timesShape[1] == 19);
}

TEST_CASE("runReproject with SeriesMode::PerFile writes one output per input file, sharing the same grid") {
    const auto dir = reprojectOutputDir("per-file");
    ReprojectOptions options;
    options.inputs = {
        "tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc",
        "tests/fixtures/reproject/wrfout_d03_2025-03-14_00_30_00.nc",
        "tests/fixtures/reproject/wrfout_d03_2025-03-14_01_00_00.nc",
    };
    options.targetEpsg = 4326;
    options.variables = {"T2"};
    options.seriesMode = SeriesMode::PerFile;
    options.outputDirectory = dir;
    const auto written = runReproject(options);
    REQUIRE(written.size() == 3);
    CHECK(written[0].filename() == "wrfout_d03_2025-03-14_00_00_00_epsg4326.nc");
    CHECK(written[1].filename() == "wrfout_d03_2025-03-14_00_30_00_epsg4326.nc");
    CHECK(written[2].filename() == "wrfout_d03_2025-03-14_01_00_00_epsg4326.nc");

    std::optional<std::vector<double>> firstLon;
    for (const auto& path : written) {
        auto out = NetcdfFile::open(path, NetcdfFile::Mode::ReadOnly);
        const auto dims = out.dimensions();
        const auto time = std::find_if(dims.begin(), dims.end(), [](const auto& d) { return d.name == "time"; });
        REQUIRE(time != dims.end());
        CHECK(time->length == 1);
        const auto lon = out.readDouble("lon");
        if (!firstLon) firstLon = lon;
        else CHECK(lon == *firstLon);  // same shared grid across every output file in the run
    }
}

TEST_CASE("runReproject rejects an empty variable list, no inputs, and a missing output directory") {
    ReprojectOptions base;
    base.inputs = {"tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc"};
    base.targetEpsg = 4326;
    base.variables = {"T2"};
    base.outputDirectory = reprojectOutputDir("errors");

    auto noVariables = base;
    noVariables.variables.clear();
    CHECK_THROWS_AS(runReproject(noVariables), UserError);

    auto noInputs = base;
    noInputs.inputs.clear();
    CHECK_THROWS_AS(runReproject(noInputs), UserError);

    auto badDirectory = base;
    badDirectory.outputDirectory = "tests/fixtures/reproject/does_not_exist";
    CHECK_THROWS_AS(runReproject(badDirectory), UserError);

    auto badVariable = base;
    badVariable.variables = {"NOT_A_REAL_VARIABLE"};
    CHECK_THROWS_AS(runReproject(badVariable), UserError);
}

TEST_CASE("runReproject to a target with a real (non-WGS84) datum, like EPSG:2326 Hong Kong 1980 Grid, lands within a "
          "pixel of the true WGS84-anchored transform, not offset by the ~300m a naive sphere->target warp gives") {
    // Regression test for a real bug found via a user report (a QGIS
    // overlay against OpenStreetMap tiles showing a visible shift): warping
    // straight from WRF's own undefined-datum modeling sphere to a target
    // CRS whose geographic base genuinely differs from WGS84 silently
    // produces PROJ's identity/no-shift fallback instead of the target's
    // real WGS84->target datum transformation - see targetSharesWgs84Datum's
    // comment in reproject.cpp for the full mechanism and the two-stage fix.
    WrfFile file("tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc");
    const auto bounds = file.geographicBounds();

    // Ground truth: the NW corner's real WGS84 lon/lat, transformed to
    // EPSG:2326 through PROJ's own published Hong Kong 1980 <-> WGS84
    // transformation - independent of this codebase's reprojection code.
    OGRSpatialReference wgs84;
    wgs84.importFromEPSG(4326);
    wgs84.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    OGRSpatialReference hk2326;
    hk2326.importFromEPSG(2326);
    hk2326.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    auto* xform = OGRCreateCoordinateTransformation(&wgs84, &hk2326);
    REQUIRE(xform);
    double expectedX = bounds.west, expectedY = bounds.north;
    const bool transformOk = xform->Transform(1, &expectedX, &expectedY);
    OGRCoordinateTransformation::DestroyCT(xform);
    REQUIRE(transformOk);

    const auto dir = reprojectOutputDir("hk1980-datum");
    ReprojectOptions options;
    options.inputs = {"tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc"};
    options.targetEpsg = 2326;
    options.variables = {"T2"};
    options.outputDirectory = dir;
    const auto written = runReproject(options);
    REQUIRE(written.size() == 1);
    auto out = NetcdfFile::open(written.front(), NetcdfFile::Mode::ReadOnly);
    const auto x = out.readDouble("x");
    const auto y = out.readDouble("y");
    REQUIRE(x.size() >= 2);
    REQUIRE(y.size() >= 2);
    // x/y are CELL CENTRES (see the writer's own comment); recover the
    // NW pixel CORNER by backing off half a pixel, matching the corner
    // `expectedX`/`expectedY` above.
    const double actualX = x.front() - (x[1] - x[0]) / 2.0;
    const double actualY = y.front() - (y[1] - y[0]) / 2.0;

    // Measured directly against both code paths while writing this test:
    // the pre-fix (single-stage, undatum-linked) warp lands ~170-222m off;
    // the fixed two-stage warp lands within ~30m (residual resampling/
    // pixel-corner noise, not the bug). 100m cleanly separates the two -
    // it must stay well below the ~170m the bug reintroduces, not just
    // "small", or a regression here would silently slip back under a
    // looser threshold the way an earlier, too-loose 250m draft of this
    // same test did.
    CHECK(std::abs(actualX - expectedX) < 100.0);
    CHECK(std::abs(actualY - expectedY) < 100.0);
}

TEST_CASE("runReproject with a derived-variables script writes LVLHT/PTOT/TK with correct shapes, attributes, and values") {
    const auto dir = reprojectOutputDir("derived-variables");
    ReprojectOptions options;
    options.inputs = {"tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc"};
    options.targetEpsg = 4326;
    options.outputDirectory = dir;
    // No pass-through options.variables at all - every output variable
    // here comes from the script, matching how ncap2 itself writes every
    // name a script assigns.
    options.derivedVariablesScript =
        "LVLHT = (( PH + PHB ) / 9.81) - HGT;\n"
        "LVLHT@units = \"m\";\n"
        "LVLHT@long_name = \"Height above ground [m]\";\n"
        "PTOT = (P + PB) * 0.01;\n"
        "PTOT@long_name = \"Total Level Pressure\";\n"
        "PTOT@units = \"mbar\";\n"
        "TK = (PTOT/1000) ^ 0.2857 * (T + 300);\n"
        "TK@units = \"Kelvin\";\n"
        "TK@long_name = \"Level Temperature [K]\";\n";
    const auto written = runReproject(options);
    REQUIRE(written.size() == 1);
    auto out = NetcdfFile::open(written.front(), NetcdfFile::Mode::ReadOnly);

    // Shapes: LVLHT stays on the RAW (undestaggered) bottom_top_stag (5
    // levels in this fixture) - unlike a pass-through-selected PH/PHB,
    // which this app would instead destagger down to 4 levels under
    // "bottom_top". PTOT/TK land on "bottom_top" since P/PB/T aren't
    // staggered at all.
    CHECK(out.variable("LVLHT").dimensionNames == std::vector<std::string>{"time", "bottom_top_stag", "lat", "lon"});
    CHECK(out.variable("PTOT").dimensionNames == std::vector<std::string>{"time", "bottom_top", "lat", "lon"});
    CHECK(out.variable("TK").dimensionNames == std::vector<std::string>{"time", "bottom_top", "lat", "lon"});
    const auto dims = out.dimensions();
    const auto stag = std::find_if(dims.begin(), dims.end(), [](const auto& d) { return d.name == "bottom_top_stag"; });
    REQUIRE(stag != dims.end());
    CHECK(stag->length == 5);
    const auto unstag = std::find_if(dims.begin(), dims.end(), [](const auto& d) { return d.name == "bottom_top"; });
    REQUIRE(unstag != dims.end());
    CHECK(unstag->length == 4);

    CHECK(out.getAttribute("LVLHT", "units").text == "m");
    CHECK(out.getAttribute("LVLHT", "long_name").text == "Height above ground [m]");
    CHECK(out.getAttribute("PTOT", "units").text == "mbar");
    CHECK(out.getAttribute("PTOT", "long_name").text == "Total Level Pressure");
    CHECK(out.getAttribute("TK", "units").text == "Kelvin");
    CHECK(out.getAttribute("TK", "long_name").text == "Level Temperature [K]");
    CHECK(out.getAttribute("LVLHT", "grid_mapping").text == "crs");

    // Every finite output pixel must fall within the SAME formula computed
    // directly from the source's own raw PH/PHB/HGT/P/PB/T, across every
    // level (bilinear resampling can only interpolate within the source's
    // own finite range, never invent values outside it - the same style of
    // check the plain pass-through T2 test above uses).
    WrfFile source("tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc");
    auto rangeOf = [](const std::vector<float>& values, float& minimum, float& maximum) {
        for (float v : values) if (std::isfinite(v)) { minimum = std::min(minimum, v); maximum = std::max(maximum, v); }
    };
    float lvlhtMin = std::numeric_limits<float>::infinity(), lvlhtMax = -std::numeric_limits<float>::infinity();
    for (int level = 0; level < 5; ++level) {
        const auto ph = source.read("PH", 0, level), phb = source.read("PHB", 0, level), hgt = source.read("HGT", 0, 0);
        std::vector<float> lvlht(ph.size());
        for (std::size_t i = 0; i < ph.size(); ++i) lvlht[i] = ((ph[i] + phb[i]) / 9.81f) - hgt[i];
        rangeOf(lvlht, lvlhtMin, lvlhtMax);
    }
    float ptotMin = std::numeric_limits<float>::infinity(), ptotMax = -std::numeric_limits<float>::infinity();
    float tkMin = std::numeric_limits<float>::infinity(), tkMax = -std::numeric_limits<float>::infinity();
    for (int level = 0; level < 4; ++level) {
        const auto p = source.read("P", 0, level), pb = source.read("PB", 0, level), t = source.read("T", 0, level);
        std::vector<float> ptot(p.size()), tk(p.size());
        for (std::size_t i = 0; i < p.size(); ++i) {
            ptot[i] = (p[i] + pb[i]) * 0.01f;
            tk[i] = std::pow(ptot[i] / 1000.0f, 0.2857f) * (t[i] + 300.0f);
        }
        rangeOf(ptot, ptotMin, ptotMax);
        rangeOf(tk, tkMin, tkMax);
    }

    const auto checkWithinRange = [](const std::vector<float>& outValues, float minimum, float maximum) {
        bool sawFinite = false;
        for (float v : outValues) {
            if (v > 1e30f) continue;
            sawFinite = true;
            CHECK(v >= minimum - std::abs(minimum) * 0.01f - 1.0f);
            CHECK(v <= maximum + std::abs(maximum) * 0.01f + 1.0f);
        }
        CHECK(sawFinite);
    };
    checkWithinRange(out.readFloat("LVLHT"), lvlhtMin, lvlhtMax);
    checkWithinRange(out.readFloat("PTOT"), ptotMin, ptotMax);
    checkWithinRange(out.readFloat("TK"), tkMin, tkMax);
}

TEST_CASE("runReproject with a derived-variables script combined with a pass-through variable shares the unstaggered dimension") {
    const auto dir = reprojectOutputDir("derived-and-passthrough");
    ReprojectOptions options;
    options.inputs = {"tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc"};
    options.targetEpsg = 4326;
    options.outputDirectory = dir;
    options.variables = {"T2"};
    options.derivedVariablesScript = "PTOT = (P + PB) * 0.01;\n";
    const auto written = runReproject(options);
    REQUIRE(written.size() == 1);
    auto out = NetcdfFile::open(written.front(), NetcdfFile::Mode::ReadOnly);
    CHECK(out.variable("T2").dimensionNames == std::vector<std::string>{"time", "lat", "lon"});
    CHECK(out.variable("PTOT").dimensionNames == std::vector<std::string>{"time", "bottom_top", "lat", "lon"});
}

TEST_CASE("runReproject: a derived variable overriding a source variable's name replaces it, even when also checked as pass-through") {
    const auto dir = reprojectOutputDir("derived-override");
    ReprojectOptions options;
    options.inputs = {"tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc"};
    options.targetEpsg = 4326;
    options.outputDirectory = dir;
    options.variables = {"T2"};  // also checked as pass-through - the override must still win, not double-define "T2"
    options.derivedVariablesScript = "T2 = T2 - 273.15;\nT2@units = \"degC\";\n";
    const auto written = runReproject(options);
    REQUIRE(written.size() == 1);
    auto out = NetcdfFile::open(written.front(), NetcdfFile::Mode::ReadOnly);

    // Exactly one T2 variable (defineVariable() would have thrown on a
    // duplicate definition if the override hadn't superseded the
    // pass-through copy).
    CHECK(out.variable("T2").dimensionNames == std::vector<std::string>{"time", "lat", "lon"});
    CHECK(out.getAttribute("T2", "units").text == "degC");  // explicit @units wins over the source's own "K"
    CHECK(out.getAttribute("T2", "long_name").text == "TEMP at 2 M");  // not set by the script - falls back to the source's own description

    WrfFile source("tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc");
    const auto sourceT2 = source.read("T2", 0, 0);
    float sourceMin = std::numeric_limits<float>::infinity(), sourceMax = -std::numeric_limits<float>::infinity();
    for (float v : sourceT2) if (std::isfinite(v)) { sourceMin = std::min(sourceMin, v); sourceMax = std::max(sourceMax, v); }
    const auto outT2 = out.readFloat("T2");
    bool sawFinite = false;
    for (float v : outT2) {
        if (v > 1e30f) continue;
        sawFinite = true;
        CHECK(v >= sourceMin - 273.15f - 0.5f);
        CHECK(v <= sourceMax - 273.15f + 0.5f);
    }
    CHECK(sawFinite);
}

TEST_CASE("runReproject: an override with no explicit @units/@long_name falls back to the source variable's own") {
    const auto dir = reprojectOutputDir("derived-override-fallback");
    ReprojectOptions options;
    options.inputs = {"tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc"};
    options.targetEpsg = 4326;
    options.outputDirectory = dir;
    options.derivedVariablesScript = "T2 = T2 + 0;\n";  // no @units/@long_name at all
    const auto written = runReproject(options);
    REQUIRE(written.size() == 1);
    auto out = NetcdfFile::open(written.front(), NetcdfFile::Mode::ReadOnly);
    CHECK(out.getAttribute("T2", "units").text == "K");
    CHECK(out.getAttribute("T2", "long_name").text == "TEMP at 2 M");
}

TEST_CASE("runReproject propagates a derived-variables script error and requires at least one output variable") {
    const auto dir = reprojectOutputDir("derived-errors");
    ReprojectOptions base;
    base.inputs = {"tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc"};
    base.targetEpsg = 4326;
    base.outputDirectory = dir;

    auto badScript = base;
    badScript.derivedVariablesScript = "X = PH + NOT_A_REAL_VARIABLE;\n";
    CHECK_THROWS_AS(runReproject(badScript), UserError);

    auto nothingSelected = base;  // no options.variables, no derivedVariablesScript
    CHECK_THROWS_AS(runReproject(nothingSelected), UserError);
}

TEST_CASE("NetcdfFile::create makes a brand-new file and writeFloatSlice/readFloatSlice round-trip a hyperslab") {
    const auto path = std::filesystem::temp_directory_path() / "wrftools-cpp-netcdf-file-create-test.nc";
    {
        auto file = NetcdfFile::create(path);
        file.defineDimension("time", 3);
        file.defineDimension("y", 2);
        file.defineDimension("x", 2);
        file.defineVariable("v", NC_FLOAT, {"time", "y", "x"});
        const std::vector<float> plane{1.0f, 2.0f, 3.0f, 4.0f};
        file.writeFloatSlice("v", {1, 0, 0}, {1, 2, 2}, plane);
    }
    {
        // Scoped so `reopened` closes before the removal below - an open
        // handle onto `path` makes std::filesystem::remove fail on Windows
        // (unlike POSIX, where an open file can be unlinked freely) - see
        // 9f90b83's identical fix for the other tests in this file.
        auto reopened = NetcdfFile::open(path, NetcdfFile::Mode::ReadOnly);
        const auto slice = reopened.readFloatSlice("v", {1, 0, 0}, {1, 2, 2});
        CHECK(slice == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
        const auto whole = reopened.readFloat("v");
        CHECK(whole.size() == 12);
        CHECK(whole[4] == 1.0f);  // (time=1, y=0, x=0)
    }
    std::filesystem::remove(path);
}

TEST_CASE("NetcdfFile::writeFloatSlice rejects a mismatched count and an out-of-bounds slice") {
    const auto path = std::filesystem::temp_directory_path() / "wrftools-cpp-netcdf-file-slice-bounds-test.nc";
    {
        // Scoped so `file` closes before the removal below - see the
        // comment on the test above.
        auto file = NetcdfFile::create(path);
        file.defineDimension("y", 2);
        file.defineDimension("x", 2);
        file.defineVariable("v", NC_FLOAT, {"y", "x"});
        CHECK_THROWS_AS(file.writeFloatSlice("v", {0, 0}, {2, 2}, std::vector<float>{1.0f}), UserError);
        CHECK_THROWS_AS(file.writeFloatSlice("v", {1, 0}, {2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}), UserError);
    }
    std::filesystem::remove(path);
}

TEST_CASE("NetcdfFile::attributes enumerates every global attribute of a real file") {
    auto file = NetcdfFile::open("tests/fixtures/reproject/wrfout_d03_2025-03-14_00_00_00.nc", NetcdfFile::Mode::ReadOnly);
    const auto globals = file.attributes("");
    const auto hasName = [&](const std::string& name) {
        return std::any_of(globals.begin(), globals.end(), [&](const auto& a) { return a.name == name; });
    };
    CHECK(hasName("MAP_PROJ"));
    CHECK(hasName("DX"));
    CHECK(hasName("TITLE"));
}

TEST_CASE("NetcdfFile::readText/writeText round-trip a Times-shaped char variable") {
    const auto path = std::filesystem::temp_directory_path() / "wrftools-cpp-netcdf-file-text-test.nc";
    {
        // Scoped so `file` closes before the removal below - see the
        // comment on the NetcdfFile::create test above.
        auto file = NetcdfFile::create(path);
        file.defineDimension("time", 2);
        file.defineDimension("DateStrLen", 19);
        file.defineVariable("Times", NC_CHAR, {"time", "DateStrLen"});
        const std::string text = "2025-03-14_00:00:002025-03-14_00:30:00";
        file.writeText("Times", text);
        CHECK(file.readText("Times") == text);
    }
    std::filesystem::remove(path);
}

TEST_CASE("computeDescriptiveStats ignores non-finite values and reports count/min/max/mean/median/stddev") {
    const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
    const auto stats = computeDescriptiveStats(values);
    CHECK(stats.count == 4);
    CHECK(stats.minimum == Catch::Approx(1.0f));
    CHECK(stats.maximum == Catch::Approx(4.0f));
    CHECK(stats.mean == Catch::Approx(2.5f));
    CHECK(stats.median == Catch::Approx(2.5f));  // even count -> average of the two middle values
    CHECK(stats.stddev == Catch::Approx(std::sqrt(1.25f)));
}

TEST_CASE("computeDescriptiveStats throws UserError when every value is non-finite") {
    const std::vector<float> values{std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
    CHECK_THROWS_AS(computeDescriptiveStats(values), UserError);
}

TEST_CASE("computeHistogram buckets values across the requested bin count") {
    const std::vector<float> values{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    const auto histogram = computeHistogram(values, 5);
    CHECK(histogram.binCounts.size() == 5);
    std::size_t total = 0;
    for (const auto count : histogram.binCounts) total += count;
    CHECK(total == values.size());
    CHECK(histogram.binStart == Catch::Approx(0.0f));
    CHECK(histogram.binWidth == Catch::Approx(2.0f));
}

TEST_CASE("computeHistogram widens a zero-range input so it still gets one visible bin") {
    const std::vector<float> values{5.0f, 5.0f, 5.0f};
    const auto histogram = computeHistogram(values, 4);
    std::size_t total = 0;
    for (const auto count : histogram.binCounts) total += count;
    CHECK(total == values.size());
    CHECK(histogram.binWidth > 0.0f);
}

namespace {
// A small resolver for derived-variable evaluate() tests: `raw` holds one
// 1-pixel array per (source variable, level); a name not found there is
// looked up among `defs` and evaluated recursively (memoized per (name,
// level) via `cache`) - the same chaining behaviour runReproject's own
// resolver closure provides for a later statement referencing an earlier
// one's name.
struct TestResolver {
    std::map<std::string, std::vector<float>> raw;
    const std::vector<DerivedVariableDef>* defs{};
    std::map<std::string, std::map<int, std::vector<float>>> cache;

    const std::vector<float>& operator()(const std::string& name, int level) {
        const auto rawKey = name + "@" + std::to_string(level);
        if (const auto found = raw.find(rawKey); found != raw.end()) return found->second;
        auto& cacheForName = cache[name];
        if (const auto cached = cacheForName.find(level); cached != cacheForName.end()) return cached->second;
        const auto found = std::find_if(defs->begin(), defs->end(), [&](const auto& d) { return d.name == name; });
        REQUIRE(found != defs->end());
        return cacheForName.emplace(level, evaluate(*found, level, std::ref(*this))).first->second;
    }
};

const std::map<std::string, VariableShape> kWrfLikeShapes{{"PH", {"bottom_top_stag", 3}}, {"PHB", {"bottom_top_stag", 3}}, {"HGT", {"", 1}},
    {"P", {"bottom_top", 2}}, {"PB", {"bottom_top", 2}}, {"T", {"bottom_top", 2}}};

constexpr const char* kMotivatingScript =
    "LVLHT = (( PH + PHB ) / 9.81) - HGT;\n"
    "LVLHT@units = \"m\";\n"
    "LVLHT@long_name = \"Height above ground [m]\";\n"
    "PTOT = (P + PB) * 0.01;\n"
    "PTOT@long_name = \"Total Level Pressure\";\n"
    "PTOT@units = \"mbar\";\n"
    "TK = (PTOT/1000) ^ 0.2857 * (T + 300);\n"
    "TK@units = \"Kelvin\";\n"
    "TK@long_name = \"Level Temperature [K]\";\n";
}  // namespace

TEST_CASE("parseDerivedVariables parses the motivating script and infers each variable's shape") {
    const auto defs = parseDerivedVariables(kMotivatingScript, kWrfLikeShapes);
    REQUIRE(defs.size() == 3);
    CHECK(defs[0].name == "LVLHT");
    CHECK(defs[0].units == "m");
    CHECK(defs[0].longName == "Height above ground [m]");
    CHECK(shapeOf(defs[0]).dimensionName == "bottom_top_stag");
    CHECK(shapeOf(defs[0]).levelCount == 3);
    CHECK(defs[1].name == "PTOT");
    CHECK(defs[1].units == "mbar");
    CHECK(defs[1].longName == "Total Level Pressure");
    CHECK(shapeOf(defs[1]).dimensionName == "bottom_top");
    CHECK(shapeOf(defs[1]).levelCount == 2);
    CHECK(defs[2].name == "TK");
    CHECK(defs[2].units == "Kelvin");
    CHECK(shapeOf(defs[2]).dimensionName == "bottom_top");
    CHECK(shapeOf(defs[2]).levelCount == 2);
}

TEST_CASE("evaluate reproduces the motivating script's formulas, including 2D broadcast and chaining") {
    const auto defs = parseDerivedVariables(kMotivatingScript, kWrfLikeShapes);
    REQUIRE(defs.size() == 3);

    TestResolver resolver;
    resolver.defs = &defs;
    resolver.raw = {{"PH@0", {1000.0f}}, {"PH@1", {2000.0f}}, {"PH@2", {3000.0f}}, {"PHB@0", {500.0f}}, {"PHB@1", {600.0f}}, {"PHB@2", {700.0f}},
        {"HGT@0", {50.0f}}, {"P@0", {100000.0f}}, {"P@1", {90000.0f}}, {"PB@0", {500.0f}}, {"PB@1", {400.0f}}, {"T@0", {10.0f}}, {"T@1", {5.0f}}};

    // LVLHT: HGT (2D) must broadcast across all 3 bottom_top_stag levels
    // even though it's only ever recorded at "HGT@0" above - if broadcast
    // weren't applied, level 1/2 would ask the resolver for a nonexistent
    // "HGT@1"/"HGT@2" and REQUIRE(found != defs->end()) above would fail.
    CHECK(evaluate(defs[0], 0, std::ref(resolver))[0] == Catch::Approx((1000.0f + 500.0f) / 9.81f - 50.0f));
    CHECK(evaluate(defs[0], 1, std::ref(resolver))[0] == Catch::Approx((2000.0f + 600.0f) / 9.81f - 50.0f));
    CHECK(evaluate(defs[0], 2, std::ref(resolver))[0] == Catch::Approx((3000.0f + 700.0f) / 9.81f - 50.0f));

    const float expectedPtot0 = (100000.0f + 500.0f) * 0.01f;
    const float expectedPtot1 = (90000.0f + 400.0f) * 0.01f;
    CHECK(evaluate(defs[1], 0, std::ref(resolver))[0] == Catch::Approx(expectedPtot0));
    CHECK(evaluate(defs[1], 1, std::ref(resolver))[0] == Catch::Approx(expectedPtot1));

    // TK references PTOT (an earlier statement in the same script) - this
    // is the chaining path, resolved recursively by TestResolver above.
    // '^' must also be right-associative and bind tighter than the
    // surrounding '*' for this to match.
    CHECK(evaluate(defs[2], 0, std::ref(resolver))[0] == Catch::Approx(std::pow(expectedPtot0 / 1000.0f, 0.2857f) * (10.0f + 300.0f)));
    CHECK(evaluate(defs[2], 1, std::ref(resolver))[0] == Catch::Approx(std::pow(expectedPtot1 / 1000.0f, 0.2857f) * (5.0f + 300.0f)));
}

TEST_CASE("comparison/logical operators and the ternary operator evaluate elementwise") {
    const std::map<std::string, VariableShape> shapes{{"T2", {"", 1}}};
    const auto defs = parseDerivedVariables("MASK = T2 > 273.15;\nCLAMPED = (T2 > 320) ? 320 : T2;\n", shapes);
    REQUIRE(defs.size() == 2);

    TestResolver resolver;
    resolver.defs = &defs;
    resolver.raw = {{"T2@0", {280.0f}}};
    CHECK(evaluate(defs[0], 0, std::ref(resolver))[0] == Catch::Approx(1.0f));
    CHECK(evaluate(defs[1], 0, std::ref(resolver))[0] == Catch::Approx(280.0f));

    TestResolver belowFreezing;
    belowFreezing.defs = &defs;
    belowFreezing.raw = {{"T2@0", {260.0f}}};
    CHECK(evaluate(defs[0], 0, std::ref(belowFreezing))[0] == Catch::Approx(0.0f));

    TestResolver hot;
    hot.defs = &defs;
    hot.raw = {{"T2@0", {330.0f}}};
    CHECK(evaluate(defs[1], 0, std::ref(hot))[0] == Catch::Approx(320.0f));
}

TEST_CASE("builtin functions match their std:: equivalents") {
    const std::map<std::string, VariableShape> shapes{{"A", {"", 1}}, {"B", {"", 1}}};
    const auto defs = parseDerivedVariables(
        "S = sqrt(A);\nPW = pow(A, B);\nMN = min(A, B);\nMX = max(A, B);\nAT2 = atan2(A, B);\n", shapes);
    REQUIRE(defs.size() == 5);

    TestResolver resolver;
    resolver.defs = &defs;
    resolver.raw = {{"A@0", {9.0f}}, {"B@0", {2.0f}}};
    CHECK(evaluate(defs[0], 0, std::ref(resolver))[0] == Catch::Approx(std::sqrt(9.0f)));
    CHECK(evaluate(defs[1], 0, std::ref(resolver))[0] == Catch::Approx(std::pow(9.0f, 2.0f)));
    CHECK(evaluate(defs[2], 0, std::ref(resolver))[0] == Catch::Approx(std::min(9.0f, 2.0f)));
    CHECK(evaluate(defs[3], 0, std::ref(resolver))[0] == Catch::Approx(std::max(9.0f, 2.0f)));
    CHECK(evaluate(defs[4], 0, std::ref(resolver))[0] == Catch::Approx(std::atan2(9.0f, 2.0f)));
}

TEST_CASE("parseDerivedVariables rejects an unknown function name and a wrong argument count") {
    const std::map<std::string, VariableShape> shapes{{"A", {"", 1}}};
    CHECK_THROWS_AS(parseDerivedVariables("X = notarealfunction(A);\n", shapes), UserError);
    CHECK_THROWS_AS(parseDerivedVariables("X = sqrt(A, A);\n", shapes), UserError);
    CHECK_THROWS_AS(parseDerivedVariables("X = pow(A);\n", shapes), UserError);
}

TEST_CASE("parseDerivedVariables rejects an unknown identifier") {
    const std::map<std::string, VariableShape> shapes{{"A", {"", 1}}};
    CHECK_THROWS_AS(parseDerivedVariables("X = A + NOPE;\n", shapes), UserError);
}

TEST_CASE("parseDerivedVariables rejects an @attr statement with no matching assignment") {
    const std::map<std::string, VariableShape> shapes{{"A", {"", 1}}};
    CHECK_THROWS_AS(parseDerivedVariables("NOPE@units = \"m\";\n", shapes), UserError);
}

TEST_CASE("parseDerivedVariables allows reassigning a source variable's own name exactly once, marking it as an override") {
    const std::map<std::string, VariableShape> shapes{{"A", {"", 1}}};
    const auto defs = parseDerivedVariables("A = A + 1;\n", shapes);
    REQUIRE(defs.size() == 1);
    CHECK(defs[0].name == "A");
    CHECK(defs[0].overridesSourceVariable);

    // The RHS's own "A" resolves to the ORIGINAL source variable, not the
    // new definition - a replacement, not infinite self-reference.
    TestResolver resolver;
    resolver.defs = &defs;
    resolver.raw = {{"A@0", {5.0f}}};
    CHECK(evaluate(defs[0], 0, std::ref(resolver))[0] == Catch::Approx(6.0f));
}

TEST_CASE("parseDerivedVariables rejects reassigning the same name twice, whether a source override or a plain derived name") {
    const std::map<std::string, VariableShape> shapes{{"A", {"", 1}}};
    CHECK_THROWS_AS(parseDerivedVariables("A = 1;\nA = 2;\n", shapes), UserError);
    CHECK_THROWS_AS(parseDerivedVariables("X = 1;\nX = 2;\n", shapes), UserError);
}

TEST_CASE("a derived variable that does not override a source variable is not marked as an override") {
    const std::map<std::string, VariableShape> shapes{{"A", {"", 1}}};
    const auto defs = parseDerivedVariables("X = A + 1;\n", shapes);
    REQUIRE(defs.size() == 1);
    CHECK_FALSE(defs[0].overridesSourceVariable);
}

TEST_CASE("parseDerivedVariables rejects combining two operands on different vertical dimensions") {
    const auto& shapes = kWrfLikeShapes;
    CHECK_THROWS_AS(parseDerivedVariables("X = PH + T;\n", shapes), UserError);          // bottom_top_stag vs bottom_top
    CHECK_THROWS_AS(parseDerivedVariables("X = min(PH, T);\n", shapes), UserError);       // same, inside a function call
}

TEST_CASE("parseDerivedVariables rejects a syntax error") {
    const std::map<std::string, VariableShape> shapes{{"A", {"", 1}}};
    CHECK_THROWS_AS(parseDerivedVariables("X = A\n", shapes), UserError);      // missing ';'
    CHECK_THROWS_AS(parseDerivedVariables("X = (A + 1;\n", shapes), UserError);  // unmatched '('
}

TEST_CASE("parseDerivedVariables ignores an empty or whitespace-only script") {
    CHECK(parseDerivedVariables("", {}).empty());
    CHECK(parseDerivedVariables("   \n\t \n", {}).empty());
}
