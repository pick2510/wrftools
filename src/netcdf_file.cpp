#include "wrftools/netcdf_file.hpp"
#include "wrftools/error.hpp"

#include <netcdf.h>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <set>
#include <utility>

namespace wrftools {
namespace {

// HDF5 (which every netCDF-4 file goes through under the hood) can hang
// indefinitely in nc_close - not just run slowly - while trying to acquire/
// release its optional SWMR-related file lock on certain filesystems;
// confirmed on GitHub Actions' own Windows runners (a real end user's
// network-mounted storage is the same class of risk), where a Reproject-tab
// export of a merged, chunked+deflate-compressed, growable-time-dimension
// output hung at exactly that point for minutes regardless of how long the
// caller was willing to wait, while an otherwise-identical fixed-size,
// uncompressed write closed instantly - HDF5_USE_FILE_LOCKING=FALSE is the
// HDF Group's own documented workaround. Set once, here, before the first
// file this class ever touches is opened or created - rather than in only
// one entry point (the Reproject worker, GUI main(), or a test binary) -
// so every caller of this class is covered, and only if the environment
// doesn't already carry an explicit value, so a caller/deployment that
// actually wants file locking can still opt back in.
void disableHdf5FileLockingIfUnset() {
#ifdef _WIN32
    if (std::getenv("HDF5_USE_FILE_LOCKING") == nullptr) _putenv_s("HDF5_USE_FILE_LOCKING", "FALSE");
#else
    setenv("HDF5_USE_FILE_LOCKING", "FALSE", 0);
#endif
}

void checkNc(int status, const std::string& what) {
    if (status != NC_NOERR) throw UserError("NetCDF error (" + what + "): " + nc_strerror(status));
}

std::size_t productOf(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
}

// Copies one variable's attribute (or, for varid == NC_GLOBAL, a global
// attribute) from one open dataset to another - a thin wrapper purely so
// callers don't have to spell out nc_copy_att's argument order themselves.
void copyAttribute(int srcNcid, int srcVarid, const std::string& name, int dstNcid, int dstVarid) {
    checkNc(nc_copy_att(srcNcid, srcVarid, name.c_str(), dstNcid, dstVarid), "nc_copy_att " + name);
}

// Reads every attribute name attached to `varid` (NC_GLOBAL for global).
std::vector<std::string> attributeNames(int ncid, int varid) {
    int count = 0;
    checkNc(nc_inq_varnatts(ncid, varid, &count), "nc_inq_varnatts");
    std::vector<std::string> names;
    names.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        char name[NC_MAX_NAME + 1] = {};
        checkNc(nc_inq_attname(ncid, varid, i, name), "nc_inq_attname");
        names.emplace_back(name);
    }
    return names;
}

// Copies one variable's full data from src to dst, dispatching on its
// on-disk type - covers every type actually seen in a WRF geo_em file
// (NC_CHAR/NC_INT/NC_FLOAT/NC_DOUBLE); anything else is rejected rather
// than silently mishandled.
//
// Uses the explicit-start/count "vara" calls, not the whole-array "var"
// ones: a variable carrying the (always first, always unlimited) Time
// dimension has a *record count*, not a fixed length - src's is whatever
// the file actually has (1, for every real geo_em file), but dst is a
// brand-new file whose Time dimension starts at 0 records. nc_put_var
// writes according to the destination's *current* dimension sizes, so
// against a 0-record Time dimension it would silently write nothing
// (confirmed the hard way: shape() came back {0, ...} and the very next
// read segfaulted on an empty buffer) - nc_put_vara with an explicit count
// sidesteps that by not depending on dst's unlimited-dimension state at
// all.
// Re-applies the source variable's shuffle/deflate settings to the freshly
// defined destination variable (netCDF-4 only - classic-format files have
// no per-variable compression, and nc_inq_var_deflate then reports
// NC_ENOTNC4). Without this, every variable rebuilt by resizeDimension/
// rebuildStructure silently came out uncompressed - a 14 MB geo_em file
// grew to ~150 MB after the LCZ conversion. Must run before nc_enddef.
// Does nothing for a source variable that wasn't compressed.
void copyCompression(int srcNcid, int srcVarid, int dstNcid, int dstVarid) {
    int shuffle = 0, deflate = 0, level = 0;
    if (nc_inq_var_deflate(srcNcid, srcVarid, &shuffle, &deflate, &level) != NC_NOERR || !deflate) return;
    checkNc(nc_def_var_deflate(dstNcid, dstVarid, shuffle, deflate, level), "nc_def_var_deflate");
}

void copyVariableData(int srcNcid, int srcVarid, NetcdfFile::NcType type, const std::vector<std::size_t>& start, const std::vector<std::size_t>& counts,
    int dstNcid, int dstVarid) {
    const std::size_t count = productOf(counts);
    switch (type) {
        case NC_CHAR: {
            std::vector<char> buffer(count);
            checkNc(nc_get_vara_text(srcNcid, srcVarid, start.data(), counts.data(), buffer.data()), "nc_get_vara_text");
            checkNc(nc_put_vara_text(dstNcid, dstVarid, start.data(), counts.data(), buffer.data()), "nc_put_vara_text");
            return;
        }
        case NC_INT: {
            std::vector<int> buffer(count);
            checkNc(nc_get_vara_int(srcNcid, srcVarid, start.data(), counts.data(), buffer.data()), "nc_get_vara_int");
            checkNc(nc_put_vara_int(dstNcid, dstVarid, start.data(), counts.data(), buffer.data()), "nc_put_vara_int");
            return;
        }
        case NC_FLOAT: {
            std::vector<float> buffer(count);
            checkNc(nc_get_vara_float(srcNcid, srcVarid, start.data(), counts.data(), buffer.data()), "nc_get_vara_float");
            checkNc(nc_put_vara_float(dstNcid, dstVarid, start.data(), counts.data(), buffer.data()), "nc_put_vara_float");
            return;
        }
        case NC_DOUBLE: {
            std::vector<double> buffer(count);
            checkNc(nc_get_vara_double(srcNcid, srcVarid, start.data(), counts.data(), buffer.data()), "nc_get_vara_double");
            checkNc(nc_put_vara_double(dstNcid, dstVarid, start.data(), counts.data(), buffer.data()), "nc_put_vara_double");
            return;
        }
        default:
            throw UnsupportedError("NetcdfFile: unsupported variable type for structural copy: " + std::to_string(type));
    }
}

}  // namespace

NetcdfFile::NetcdfFile(int ncid, std::filesystem::path path, Mode mode) : ncid_(ncid), path_(std::move(path)), mode_(mode) {}

NetcdfFile NetcdfFile::open(const std::filesystem::path& path, Mode mode) {
    disableHdf5FileLockingIfUnset();
    int ncid = -1;
    checkNc(nc_open(path.string().c_str(), mode == Mode::ReadWrite ? NC_WRITE : NC_NOWRITE, &ncid), "nc_open " + path.string());
    return NetcdfFile(ncid, path, mode);
}

NetcdfFile NetcdfFile::create(const std::filesystem::path& path, Format format) {
    disableHdf5FileLockingIfUnset();
    int createMode = NC_CLOBBER;
    switch (format) {
        case Format::Netcdf4: createMode |= NC_NETCDF4; break;
        case Format::Netcdf4Classic: createMode |= NC_NETCDF4 | NC_CLASSIC_MODEL; break;
        case Format::Classic: break;
    }
    int ncid = -1;
    checkNc(nc_create(path.string().c_str(), createMode, &ncid), "nc_create " + path.string());
    return NetcdfFile(ncid, path, Mode::ReadWrite);
}

NetcdfFile::~NetcdfFile() { close(); }

NetcdfFile::NetcdfFile(NetcdfFile&& other) noexcept : ncid_(other.ncid_), path_(std::move(other.path_)), mode_(other.mode_) { other.ncid_ = -1; }

NetcdfFile& NetcdfFile::operator=(NetcdfFile&& other) noexcept {
    if (this != &other) {
        close();
        ncid_ = other.ncid_;
        path_ = std::move(other.path_);
        mode_ = other.mode_;
        other.ncid_ = -1;
    }
    return *this;
}

void NetcdfFile::close() {
    if (ncid_ != -1) {
        nc_close(ncid_);  // deliberately not checkNc'd - a destructor-reachable path shouldn't throw
        ncid_ = -1;
    }
}

std::vector<NetcdfFile::Dimension> NetcdfFile::dimensions() const {
    int ndims = 0;
    checkNc(nc_inq_ndims(ncid_, &ndims), "nc_inq_ndims");
    int unlimDimid = -1;
    checkNc(nc_inq_unlimdim(ncid_, &unlimDimid), "nc_inq_unlimdim");

    std::vector<Dimension> result;
    result.reserve(static_cast<std::size_t>(ndims));
    for (int dimid = 0; dimid < ndims; ++dimid) {
        char name[NC_MAX_NAME + 1] = {};
        std::size_t length = 0;
        checkNc(nc_inq_dim(ncid_, dimid, name, &length), "nc_inq_dim");
        result.push_back({name, length, dimid == unlimDimid});
    }
    return result;
}

std::vector<NetcdfFile::Variable> NetcdfFile::variables() const {
    int nvars = 0;
    checkNc(nc_inq_nvars(ncid_, &nvars), "nc_inq_nvars");

    std::vector<Variable> result;
    result.reserve(static_cast<std::size_t>(nvars));
    for (int varid = 0; varid < nvars; ++varid) {
        char name[NC_MAX_NAME + 1] = {};
        NcType type = 0;
        int ndims = 0;
        checkNc(nc_inq_var(ncid_, varid, name, &type, &ndims, nullptr, nullptr), "nc_inq_var");
        std::vector<int> dimids(static_cast<std::size_t>(ndims));
        checkNc(nc_inq_vardimid(ncid_, varid, dimids.data()), "nc_inq_vardimid");

        std::vector<std::string> dimensionNames;
        dimensionNames.reserve(dimids.size());
        for (int dimid : dimids) {
            char dimName[NC_MAX_NAME + 1] = {};
            checkNc(nc_inq_dimname(ncid_, dimid, dimName), "nc_inq_dimname");
            dimensionNames.emplace_back(dimName);
        }
        result.push_back({name, type, std::move(dimensionNames)});
    }
    return result;
}

bool NetcdfFile::hasVariable(const std::string& name) const {
    int varid = -1;
    return nc_inq_varid(ncid_, name.c_str(), &varid) == NC_NOERR;
}

NetcdfFile::Variable NetcdfFile::variable(const std::string& name) const {
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, name.c_str(), &varid), "nc_inq_varid " + name);
    char actualName[NC_MAX_NAME + 1] = {};
    NcType type = 0;
    int ndims = 0;
    checkNc(nc_inq_var(ncid_, varid, actualName, &type, &ndims, nullptr, nullptr), "nc_inq_var " + name);
    std::vector<int> dimids(static_cast<std::size_t>(ndims));
    checkNc(nc_inq_vardimid(ncid_, varid, dimids.data()), "nc_inq_vardimid " + name);
    std::vector<std::string> dimensionNames;
    dimensionNames.reserve(dimids.size());
    for (int dimid : dimids) {
        char dimName[NC_MAX_NAME + 1] = {};
        checkNc(nc_inq_dimname(ncid_, dimid, dimName), "nc_inq_dimname");
        dimensionNames.emplace_back(dimName);
    }
    return {actualName, type, std::move(dimensionNames)};
}

std::vector<std::size_t> NetcdfFile::shape(const std::string& variableName) const {
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
    int ndims = 0;
    checkNc(nc_inq_varndims(ncid_, varid, &ndims), "nc_inq_varndims");
    std::vector<int> dimids(static_cast<std::size_t>(ndims));
    checkNc(nc_inq_vardimid(ncid_, varid, dimids.data()), "nc_inq_vardimid");
    std::vector<std::size_t> result;
    result.reserve(dimids.size());
    for (int dimid : dimids) {
        std::size_t length = 0;
        checkNc(nc_inq_dimlen(ncid_, dimid, &length), "nc_inq_dimlen");
        result.push_back(length);
    }
    return result;
}

namespace {
int varidFor(int ncid, const std::string& variableName) {
    return variableName.empty() ? NC_GLOBAL : [&] {
        int varid = -1;
        checkNc(nc_inq_varid(ncid, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
        return varid;
    }();
}
}  // namespace

bool NetcdfFile::hasAttribute(const std::string& variableName, const std::string& attrName) const {
    const int varid = variableName.empty() ? NC_GLOBAL : [&] {
        int id = -1;
        return nc_inq_varid(ncid_, variableName.c_str(), &id) == NC_NOERR ? id : -2;
    }();
    if (varid == -2) return false;
    return nc_inq_att(ncid_, varid, attrName.c_str(), nullptr, nullptr) == NC_NOERR;
}

NetcdfFile::Attribute NetcdfFile::getAttribute(const std::string& variableName, const std::string& attrName) const {
    const int varid = varidFor(ncid_, variableName);
    NcType type = 0;
    std::size_t length = 0;
    checkNc(nc_inq_att(ncid_, varid, attrName.c_str(), &type, &length), "nc_inq_att " + attrName);

    Attribute attribute;
    attribute.name = attrName;
    attribute.type = type;
    if (type == NC_CHAR) {
        std::string text(length, '\0');
        if (length > 0) checkNc(nc_get_att_text(ncid_, varid, attrName.c_str(), text.data()), "nc_get_att_text " + attrName);
        attribute.text = std::move(text);
    } else {
        std::vector<double> numbers(length);
        if (length > 0) checkNc(nc_get_att_double(ncid_, varid, attrName.c_str(), numbers.data()), "nc_get_att_double " + attrName);
        attribute.numbers = std::move(numbers);
    }
    return attribute;
}

std::vector<NetcdfFile::Attribute> NetcdfFile::attributes(const std::string& variableName) const {
    const int varid = varidFor(ncid_, variableName);
    std::vector<Attribute> result;
    for (const auto& name : attributeNames(ncid_, varid)) result.push_back(getAttribute(variableName, name));
    return result;
}

void NetcdfFile::putAttribute(const std::string& variableName, const Attribute& attribute) {
    if (mode_ != Mode::ReadWrite) throw UserError("NetcdfFile::putAttribute: file was not opened read-write: " + path_.string());
    const int varid = varidFor(ncid_, variableName);

    const int redefStatus = nc_redef(ncid_);
    if (redefStatus != NC_NOERR && redefStatus != NC_EINDEFINE) checkNc(redefStatus, "nc_redef");

    if (attribute.type == NC_CHAR) {
        checkNc(nc_put_att_text(ncid_, varid, attribute.name.c_str(), attribute.text.size(), attribute.text.data()), "nc_put_att_text " + attribute.name);
    } else {
        checkNc(nc_put_att_double(ncid_, varid, attribute.name.c_str(), attribute.type, attribute.numbers.size(), attribute.numbers.data()),
            "nc_put_att_double " + attribute.name);
    }

    checkNc(nc_enddef(ncid_), "nc_enddef");
}

void NetcdfFile::defineDimension(const std::string& name, std::size_t length) {
    if (mode_ != Mode::ReadWrite) throw UserError("NetcdfFile::defineDimension: file was not opened read-write: " + path_.string());
    const int redefStatus = nc_redef(ncid_);
    if (redefStatus != NC_NOERR && redefStatus != NC_EINDEFINE) checkNc(redefStatus, "nc_redef");
    int dimid = -1;
    checkNc(nc_def_dim(ncid_, name.c_str(), length, &dimid), "nc_def_dim " + name);
    checkNc(nc_enddef(ncid_), "nc_enddef");
}

void NetcdfFile::defineUnlimitedDimension(const std::string& name) {
    if (mode_ != Mode::ReadWrite) throw UserError("NetcdfFile::defineUnlimitedDimension: file was not opened read-write: " + path_.string());
    const int redefStatus = nc_redef(ncid_);
    if (redefStatus != NC_NOERR && redefStatus != NC_EINDEFINE) checkNc(redefStatus, "nc_redef");
    int dimid = -1;
    checkNc(nc_def_dim(ncid_, name.c_str(), NC_UNLIMITED, &dimid), "nc_def_dim " + name);
    checkNc(nc_enddef(ncid_), "nc_enddef");
}

void NetcdfFile::defineVariable(const std::string& name, NcType type, const std::vector<std::string>& dimensionNames,
    const std::vector<std::size_t>& chunkSizes, int deflateLevel, std::optional<float> fillValue) {
    if (mode_ != Mode::ReadWrite) throw UserError("NetcdfFile::defineVariable: file was not opened read-write: " + path_.string());
    std::vector<int> dimids;
    dimids.reserve(dimensionNames.size());
    for (const auto& dimName : dimensionNames) {
        int dimid = -1;
        checkNc(nc_inq_dimid(ncid_, dimName.c_str(), &dimid), "nc_inq_dimid " + dimName);
        dimids.push_back(dimid);
    }
    const int redefStatus = nc_redef(ncid_);
    if (redefStatus != NC_NOERR && redefStatus != NC_EINDEFINE) checkNc(redefStatus, "nc_redef");
    int varid = -1;
    checkNc(nc_def_var(ncid_, name.c_str(), type, static_cast<int>(dimids.size()), dimids.data(), &varid), "nc_def_var " + name);

    int format = 0;
    checkNc(nc_inq_format(ncid_, &format), "nc_inq_format");
    if (format == NC_FORMAT_NETCDF4 || format == NC_FORMAT_NETCDF4_CLASSIC) {
        if (!chunkSizes.empty()) checkNc(nc_def_var_chunking(ncid_, varid, NC_CHUNKED, chunkSizes.data()), "nc_def_var_chunking " + name);
        if (deflateLevel >= 0) checkNc(nc_def_var_deflate(ncid_, varid, /*shuffle=*/0, /*deflate=*/1, deflateLevel), "nc_def_var_deflate " + name);
    }
    if (fillValue) {
        const float value = *fillValue;
        checkNc(nc_put_att_float(ncid_, varid, "_FillValue", type, 1, &value), "nc_put_att_float _FillValue " + name);
    }
    checkNc(nc_enddef(ncid_), "nc_enddef");
}

std::string NetcdfFile::readText(const std::string& variableName) const {
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
    std::string data(productOf(shape(variableName)), '\0');
    if (!data.empty()) checkNc(nc_get_var_text(ncid_, varid, data.data()), "nc_get_var_text " + variableName);
    return data;
}

void NetcdfFile::writeText(const std::string& variableName, const std::string& data) {
    if (mode_ != Mode::ReadWrite) throw UserError("NetcdfFile::writeText: file was not opened read-write: " + path_.string());
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
    const auto expected = productOf(shape(variableName));
    if (data.size() != expected)
        throw UserError("NetcdfFile: data size " + std::to_string(data.size()) + " does not match variable " + variableName + "'s shape (" +
                         std::to_string(expected) + " elements)");
    checkNc(nc_put_var_text(ncid_, varid, data.data()), "nc_put_var_text " + variableName);
}

namespace {
// `shape[i]` for an unlimited dimension is the file's CURRENT record count,
// which nc_put_vara_* is free to grow past (that's the entire point of an
// unlimited dimension) - so that one dimension is exempted from the
// out-of-bounds check below, both for reads (an out-of-range read against
// it still fails inside netCDF-C itself, with a netCDF error) and writes.
void checkSliceShape(const std::string& variableName, const std::vector<std::size_t>& shape, bool firstDimensionUnlimited,
    const std::vector<std::size_t>& start, const std::vector<std::size_t>& count, std::size_t dataSize) {
    if (start.size() != shape.size() || count.size() != shape.size())
        throw UserError("NetcdfFile: start/count rank does not match variable " + variableName + "'s rank (" + std::to_string(shape.size()) + ")");
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (firstDimensionUnlimited && i == 0) continue;
        if (start[i] + count[i] > shape[i])
            throw UserError("NetcdfFile: slice out of bounds for variable " + variableName + " on dimension " + std::to_string(i));
    }
    if (dataSize != productOf(count))
        throw UserError("NetcdfFile: data size " + std::to_string(dataSize) + " does not match requested slice (" + std::to_string(productOf(count)) +
                         " elements) for variable " + variableName);
}

bool variableStartsWithUnlimitedDimension(int ncid, int varid) {
    int unlimDimid = -1;
    checkNc(nc_inq_unlimdim(ncid, &unlimDimid), "nc_inq_unlimdim");
    if (unlimDimid == -1) return false;
    int ndims = 0;
    checkNc(nc_inq_varndims(ncid, varid, &ndims), "nc_inq_varndims");
    if (ndims == 0) return false;
    std::vector<int> dimids(static_cast<std::size_t>(ndims));
    checkNc(nc_inq_vardimid(ncid, varid, dimids.data()), "nc_inq_vardimid");
    return dimids.front() == unlimDimid;
}
}  // namespace

std::vector<float> NetcdfFile::readFloatSlice(const std::string& variableName, const std::vector<std::size_t>& start, const std::vector<std::size_t>& count) const {
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
    checkSliceShape(variableName, shape(variableName), variableStartsWithUnlimitedDimension(ncid_, varid), start, count, productOf(count));
    std::vector<float> data(productOf(count));
    if (!data.empty()) checkNc(nc_get_vara_float(ncid_, varid, start.data(), count.data(), data.data()), "nc_get_vara_float " + variableName);
    return data;
}

void NetcdfFile::writeFloatSlice(const std::string& variableName, const std::vector<std::size_t>& start, const std::vector<std::size_t>& count, const std::vector<float>& data) {
    if (mode_ != Mode::ReadWrite) throw UserError("NetcdfFile::writeFloatSlice: file was not opened read-write: " + path_.string());
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
    checkSliceShape(variableName, shape(variableName), variableStartsWithUnlimitedDimension(ncid_, varid), start, count, data.size());
    checkNc(nc_put_vara_float(ncid_, varid, start.data(), count.data(), data.data()), "nc_put_vara_float " + variableName);
}

void NetcdfFile::writeDoubleSlice(const std::string& variableName, const std::vector<std::size_t>& start, const std::vector<std::size_t>& count, const std::vector<double>& data) {
    if (mode_ != Mode::ReadWrite) throw UserError("NetcdfFile::writeDoubleSlice: file was not opened read-write: " + path_.string());
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
    checkSliceShape(variableName, shape(variableName), variableStartsWithUnlimitedDimension(ncid_, varid), start, count, data.size());
    checkNc(nc_put_vara_double(ncid_, varid, start.data(), count.data(), data.data()), "nc_put_vara_double " + variableName);
}

std::vector<float> NetcdfFile::readFloat(const std::string& variableName) const {
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
    std::vector<float> data(productOf(shape(variableName)));
    if (!data.empty()) checkNc(nc_get_var_float(ncid_, varid, data.data()), "nc_get_var_float " + variableName);
    return data;
}

std::vector<double> NetcdfFile::readDouble(const std::string& variableName) const {
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
    std::vector<double> data(productOf(shape(variableName)));
    if (!data.empty()) checkNc(nc_get_var_double(ncid_, varid, data.data()), "nc_get_var_double " + variableName);
    return data;
}

std::vector<std::int32_t> NetcdfFile::readInt(const std::string& variableName) const {
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
    std::vector<std::int32_t> data(productOf(shape(variableName)));
    if (!data.empty()) checkNc(nc_get_var_int(ncid_, varid, data.data()), "nc_get_var_int " + variableName);
    return data;
}

namespace {
void checkWritableShape(const std::string& variableName, std::size_t dataSize, std::size_t expected) {
    if (dataSize != expected)
        throw UserError("NetcdfFile: data size " + std::to_string(dataSize) + " does not match variable " + variableName + "'s shape (" +
                         std::to_string(expected) + " elements)");
}
}  // namespace

void NetcdfFile::writeFloat(const std::string& variableName, const std::vector<float>& data) {
    if (mode_ != Mode::ReadWrite) throw UserError("NetcdfFile::writeFloat: file was not opened read-write: " + path_.string());
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
    checkWritableShape(variableName, data.size(), productOf(shape(variableName)));
    checkNc(nc_put_var_float(ncid_, varid, data.data()), "nc_put_var_float " + variableName);
}

void NetcdfFile::writeDouble(const std::string& variableName, const std::vector<double>& data) {
    if (mode_ != Mode::ReadWrite) throw UserError("NetcdfFile::writeDouble: file was not opened read-write: " + path_.string());
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
    checkWritableShape(variableName, data.size(), productOf(shape(variableName)));
    checkNc(nc_put_var_double(ncid_, varid, data.data()), "nc_put_var_double " + variableName);
}

void NetcdfFile::writeInt(const std::string& variableName, const std::vector<std::int32_t>& data) {
    if (mode_ != Mode::ReadWrite) throw UserError("NetcdfFile::writeInt: file was not opened read-write: " + path_.string());
    int varid = -1;
    checkNc(nc_inq_varid(ncid_, variableName.c_str(), &varid), "nc_inq_varid " + variableName);
    checkWritableShape(variableName, data.size(), productOf(shape(variableName)));
    checkNc(nc_put_var_int(ncid_, varid, data.data()), "nc_put_var_int " + variableName);
}

void NetcdfFile::copyFile(const std::filesystem::path& src, const std::filesystem::path& dst) {
    std::error_code error;
    std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing, error);
    if (error) throw UserError("Could not copy " + src.string() + " to " + dst.string() + ": " + error.message());
}

void NetcdfFile::resizeDimension(const std::filesystem::path& path, const std::string& dimensionName, std::size_t newLength,
    const std::function<std::optional<std::vector<float>>(const std::string& variableName)>& newData) {
    int srcNcid = -1;
    checkNc(nc_open(path.string().c_str(), NC_NOWRITE, &srcNcid), "nc_open " + path.string());

    int format = 0;
    checkNc(nc_inq_format(srcNcid, &format), "nc_inq_format");
    int createMode = NC_CLOBBER;
    switch (format) {
        case NC_FORMAT_NETCDF4: createMode |= NC_NETCDF4; break;
        case NC_FORMAT_NETCDF4_CLASSIC: createMode |= NC_NETCDF4 | NC_CLASSIC_MODEL; break;
        case NC_FORMAT_64BIT_OFFSET: createMode |= NC_64BIT_OFFSET; break;
        case NC_FORMAT_64BIT_DATA: createMode |= NC_64BIT_DATA; break;
        default: break;  // classic
    }

    const auto tmpPath = path.string() + ".resize_tmp";
    int dstNcid = -1;
    checkNc(nc_create(tmpPath.c_str(), createMode, &dstNcid), "nc_create " + tmpPath);

    try {
        // Global attributes.
        for (const auto& name : attributeNames(srcNcid, NC_GLOBAL)) copyAttribute(srcNcid, NC_GLOBAL, name, dstNcid, NC_GLOBAL);

        // Dimensions - dimension ids are assigned in definition order
        // starting at 0 on both sides since both loops run 0..ndims-1 in
        // the same order, so no explicit id remapping is needed.
        int ndims = 0;
        checkNc(nc_inq_ndims(srcNcid, &ndims), "nc_inq_ndims");
        int unlimDimid = -1;
        checkNc(nc_inq_unlimdim(srcNcid, &unlimDimid), "nc_inq_unlimdim");
        for (int dimid = 0; dimid < ndims; ++dimid) {
            char name[NC_MAX_NAME + 1] = {};
            std::size_t length = 0;
            checkNc(nc_inq_dim(srcNcid, dimid, name, &length), "nc_inq_dim");
            const bool isUnlimited = dimid == unlimDimid;
            const std::size_t effectiveLength = (std::string(name) == dimensionName) ? newLength : length;
            int newDimId = -1;
            checkNc(nc_def_dim(dstNcid, name, isUnlimited ? NC_UNLIMITED : effectiveLength, &newDimId), "nc_def_dim " + std::string(name));
            if (newDimId != dimid) throw UserError("NetcdfFile::resizeDimension: unexpected dimension id mismatch copying " + path.string());
        }

        // Variables (definitions + attributes only - data copied below,
        // after nc_enddef).
        int nvars = 0;
        checkNc(nc_inq_nvars(srcNcid, &nvars), "nc_inq_nvars");
        for (int varid = 0; varid < nvars; ++varid) {
            char name[NC_MAX_NAME + 1] = {};
            NcType type = 0;
            int varNdims = 0;
            checkNc(nc_inq_var(srcNcid, varid, name, &type, &varNdims, nullptr, nullptr), "nc_inq_var");
            std::vector<int> dimids(static_cast<std::size_t>(varNdims));
            checkNc(nc_inq_vardimid(srcNcid, varid, dimids.data()), "nc_inq_vardimid");

            int newVarId = -1;
            checkNc(nc_def_var(dstNcid, name, type, varNdims, dimids.data(), &newVarId), "nc_def_var " + std::string(name));
            if (newVarId != varid) throw UserError("NetcdfFile::resizeDimension: unexpected variable id mismatch copying " + path.string());
            copyCompression(srcNcid, varid, dstNcid, newVarId);

            for (const auto& attrName : attributeNames(srcNcid, varid)) copyAttribute(srcNcid, varid, attrName, dstNcid, newVarId);
        }

        checkNc(nc_enddef(dstNcid), "nc_enddef");

        // Data: variables untouched by the resize are byte-copied;
        // variables that use the resized dimension get `newData`'s value
        // (or are left at netCDF's fill value if it returns nullopt).
        for (int varid = 0; varid < nvars; ++varid) {
            char name[NC_MAX_NAME + 1] = {};
            NcType type = 0;
            int varNdims = 0;
            checkNc(nc_inq_var(srcNcid, varid, name, &type, &varNdims, nullptr, nullptr), "nc_inq_var");
            std::vector<int> dimids(static_cast<std::size_t>(varNdims));
            checkNc(nc_inq_vardimid(srcNcid, varid, dimids.data()), "nc_inq_vardimid");

            bool usesResizedDimension = false;
            for (int dimid : dimids) {
                char dimName[NC_MAX_NAME + 1] = {};
                checkNc(nc_inq_dimname(srcNcid, dimid, dimName), "nc_inq_dimname");
                if (std::string(dimName) == dimensionName) {
                    usesResizedDimension = true;
                    break;
                }
            }

            // Per-dimension counts as they actually stand in the SOURCE
            // file (in particular, the unlimited Time dimension's current
            // record count, not NC_UNLIMITED/0) - see copyVariableData's
            // comment for why this must never be read off `dstNcid`.
            std::vector<std::size_t> counts;
            counts.reserve(dimids.size());
            for (int dimid : dimids) {
                std::size_t length = 0;
                checkNc(nc_inq_dimlen(srcNcid, dimid, &length), "nc_inq_dimlen");
                counts.push_back(length);
            }
            const std::vector<std::size_t> start(dimids.size(), 0);

            if (!usesResizedDimension) {
                copyVariableData(srcNcid, varid, type, start, counts, dstNcid, varid);
                continue;
            }

            if (const auto replacement = newData(name)) {
                // The resized dimension's count must reflect the NEW
                // length here (unlike the untouched-variable branch above,
                // which mirrors the source as-is).
                std::vector<std::size_t> newCounts = counts;
                for (std::size_t i = 0; i < dimids.size(); ++i) {
                    char dimName[NC_MAX_NAME + 1] = {};
                    checkNc(nc_inq_dimname(srcNcid, dimids[i], dimName), "nc_inq_dimname");
                    if (std::string(dimName) == dimensionName) newCounts[i] = newLength;
                }
                checkNc(nc_put_vara_float(dstNcid, varid, start.data(), newCounts.data(), replacement->data()), "nc_put_vara_float " + std::string(name));
            }
        }
    } catch (...) {
        nc_close(srcNcid);
        nc_close(dstNcid);
        std::error_code ignored;
        std::filesystem::remove(tmpPath, ignored);
        throw;
    }

    checkNc(nc_close(srcNcid), "nc_close src");
    checkNc(nc_close(dstNcid), "nc_close dst");

    std::error_code error;
    std::filesystem::remove(path, error);  // ignore: path may legitimately not exist as a leftover, rename below is what must succeed
    std::filesystem::rename(tmpPath, path, error);
    if (error) throw UserError("Could not replace " + path.string() + " with resized version: " + error.message());
}

void NetcdfFile::rebuildStructure(const std::filesystem::path& path, const std::string& resizedDimensionName, std::size_t resizedDimensionNewLength,
    const std::function<std::optional<std::vector<float>>(const std::string& variableName)>& resizedDimensionNewData,
    const std::vector<std::pair<std::string, std::size_t>>& newDimensions, const std::vector<VariableOverride>& variableOverrides) {
    int srcNcid = -1;
    checkNc(nc_open(path.string().c_str(), NC_NOWRITE, &srcNcid), "nc_open " + path.string());

    int format = 0;
    checkNc(nc_inq_format(srcNcid, &format), "nc_inq_format");
    int createMode = NC_CLOBBER;
    switch (format) {
        case NC_FORMAT_NETCDF4: createMode |= NC_NETCDF4; break;
        case NC_FORMAT_NETCDF4_CLASSIC: createMode |= NC_NETCDF4 | NC_CLASSIC_MODEL; break;
        case NC_FORMAT_64BIT_OFFSET: createMode |= NC_64BIT_OFFSET; break;
        case NC_FORMAT_64BIT_DATA: createMode |= NC_64BIT_DATA; break;
        default: break;  // classic
    }

    const auto tmpPath = path.string() + ".rebuild_tmp";
    int dstNcid = -1;
    checkNc(nc_create(tmpPath.c_str(), createMode, &dstNcid), "nc_create " + tmpPath);

    auto overrideFor = [&](const std::string& name) -> const VariableOverride* {
        for (const auto& override : variableOverrides)
            if (override.name == name) return &override;
        return nullptr;
    };

    try {
        for (const auto& name : attributeNames(srcNcid, NC_GLOBAL)) copyAttribute(srcNcid, NC_GLOBAL, name, dstNcid, NC_GLOBAL);

        // A `newDimensions` entry is usually genuinely new (e.g.
        // "num_urb_params" on a source file that never had urban physics
        // configured at all), but some real-world geo_em files already
        // carry a dimension of that exact name themselves - e.g. from a
        // newer WRF/geogrid version's own default urban physics scheme, or
        // from a file this tool (or w2w itself) already processed once.
        // Treat any such collision as a resize of the EXISTING dimension
        // (same handling as resizedDimensionName below), not a fresh
        // nc_def_dim - which would otherwise fail with "String match to
        // name in use" the moment a real file already has it.
        std::set<std::string> newDimensionNamesAlreadyInSource;
        for (const auto& [name, length] : newDimensions) {
            int existingDimid = -1;
            if (nc_inq_dimid(srcNcid, name.c_str(), &existingDimid) == NC_NOERR) newDimensionNamesAlreadyInSource.insert(name);
        }
        const auto resizedLengthFor = [&](const std::string& name) -> std::optional<std::size_t> {
            if (name == resizedDimensionName) return resizedDimensionNewLength;
            for (const auto& [newName, newLength] : newDimensions)
                if (newName == name && newDimensionNamesAlreadyInSource.contains(name)) return newLength;
            return std::nullopt;
        };

        // Existing dimensions - ids assigned in definition order on both
        // sides, as in resizeDimension.
        int ndims = 0;
        checkNc(nc_inq_ndims(srcNcid, &ndims), "nc_inq_ndims");
        int unlimDimid = -1;
        checkNc(nc_inq_unlimdim(srcNcid, &unlimDimid), "nc_inq_unlimdim");
        for (int dimid = 0; dimid < ndims; ++dimid) {
            char name[NC_MAX_NAME + 1] = {};
            std::size_t length = 0;
            checkNc(nc_inq_dim(srcNcid, dimid, name, &length), "nc_inq_dim");
            const bool isUnlimited = dimid == unlimDimid;
            const auto resized = resizedLengthFor(name);
            const std::size_t effectiveLength = resized.value_or(length);
            int newDimId = -1;
            checkNc(nc_def_dim(dstNcid, name, isUnlimited ? NC_UNLIMITED : effectiveLength, &newDimId), "nc_def_dim " + std::string(name));
            if (newDimId != dimid) throw UserError("NetcdfFile::rebuildStructure: unexpected dimension id mismatch copying " + path.string());
        }
        // Brand-new dimensions the overrides need - skipping any that
        // already existed in the source file and were resized in place
        // above.
        for (const auto& [name, length] : newDimensions) {
            if (newDimensionNamesAlreadyInSource.contains(name)) continue;
            int newDimId = -1;
            checkNc(nc_def_dim(dstNcid, name.c_str(), length, &newDimId), "nc_def_dim " + name);
        }

        // Variables: an overridden name is defined fresh (its own type/
        // dims, no attributes copied - callers add whatever attributes
        // they need afterward via putAttribute); everything else is
        // copied unchanged, exactly as resizeDimension does.
        int nvars = 0;
        checkNc(nc_inq_nvars(srcNcid, &nvars), "nc_inq_nvars");
        std::vector<bool> isOverridden(static_cast<std::size_t>(nvars), false);
        for (int varid = 0; varid < nvars; ++varid) {
            char name[NC_MAX_NAME + 1] = {};
            NcType type = 0;
            int varNdims = 0;
            checkNc(nc_inq_var(srcNcid, varid, name, &type, &varNdims, nullptr, nullptr), "nc_inq_var");

            if (const auto* override = overrideFor(name)) {
                isOverridden[static_cast<std::size_t>(varid)] = true;
                std::vector<int> newDimids;
                newDimids.reserve(override->dimensionNames.size());
                for (const auto& dimName : override->dimensionNames) {
                    int dimid = -1;
                    checkNc(nc_inq_dimid(dstNcid, dimName.c_str(), &dimid), "nc_inq_dimid " + dimName);
                    newDimids.push_back(dimid);
                }
                int newVarId = -1;
                checkNc(nc_def_var(dstNcid, name, override->type, static_cast<int>(newDimids.size()), newDimids.data(), &newVarId),
                    "nc_def_var (override) " + std::string(name));
                if (newVarId != varid) throw UserError("NetcdfFile::rebuildStructure: unexpected variable id mismatch copying " + path.string());
                copyCompression(srcNcid, varid, dstNcid, newVarId);  // the overridden variable's own source counterpart (same varid)
                continue;
            }

            std::vector<int> dimids(static_cast<std::size_t>(varNdims));
            checkNc(nc_inq_vardimid(srcNcid, varid, dimids.data()), "nc_inq_vardimid");
            int newVarId = -1;
            checkNc(nc_def_var(dstNcid, name, type, varNdims, dimids.data(), &newVarId), "nc_def_var " + std::string(name));
            if (newVarId != varid) throw UserError("NetcdfFile::rebuildStructure: unexpected variable id mismatch copying " + path.string());
            copyCompression(srcNcid, varid, dstNcid, newVarId);
            for (const auto& attrName : attributeNames(srcNcid, varid)) copyAttribute(srcNcid, varid, attrName, dstNcid, newVarId);
        }

        checkNc(nc_enddef(dstNcid), "nc_enddef");

        for (int varid = 0; varid < nvars; ++varid) {
            char name[NC_MAX_NAME + 1] = {};
            NcType type = 0;
            int varNdims = 0;
            checkNc(nc_inq_var(srcNcid, varid, name, &type, &varNdims, nullptr, nullptr), "nc_inq_var");

            if (isOverridden[static_cast<std::size_t>(varid)]) {
                // nc_put_var (whole-array) writes according to the
                // destination's CURRENT unlimited-dimension record count,
                // which for a variable's Time dimension may still be 0 at
                // this point regardless of write order elsewhere in this
                // loop (nothing guarantees another variable already
                // extended it first) - see copyVariableData's comment for
                // the same hazard. Use explicit vara with count=1 for
                // Time instead, matching every other writer in this file.
                const auto* override = overrideFor(name);
                int unlimDimidDst = -1;
                checkNc(nc_inq_unlimdim(dstNcid, &unlimDimidDst), "nc_inq_unlimdim");
                std::vector<int> newDimids;
                newDimids.reserve(override->dimensionNames.size());
                for (const auto& dimName : override->dimensionNames) {
                    int dimid = -1;
                    checkNc(nc_inq_dimid(dstNcid, dimName.c_str(), &dimid), "nc_inq_dimid " + dimName);
                    newDimids.push_back(dimid);
                }
                std::vector<std::size_t> counts(newDimids.size());
                for (std::size_t i = 0; i < newDimids.size(); ++i) {
                    if (newDimids[i] == unlimDimidDst) {
                        counts[i] = 1;
                    } else {
                        std::size_t length = 0;
                        checkNc(nc_inq_dimlen(dstNcid, newDimids[i], &length), "nc_inq_dimlen");
                        counts[i] = length;
                    }
                }
                const std::vector<std::size_t> start(newDimids.size(), 0);
                checkNc(nc_put_vara_float(dstNcid, varid, start.data(), counts.data(), override->data.data()), "nc_put_vara_float (override) " + std::string(name));
                continue;
            }

            std::vector<int> dimids(static_cast<std::size_t>(varNdims));
            checkNc(nc_inq_vardimid(srcNcid, varid, dimids.data()), "nc_inq_vardimid");

            bool usesResizedDimension = false;
            for (int dimid : dimids) {
                char dimName[NC_MAX_NAME + 1] = {};
                checkNc(nc_inq_dimname(srcNcid, dimid, dimName), "nc_inq_dimname");
                if (std::string(dimName) == resizedDimensionName) {
                    usesResizedDimension = true;
                    break;
                }
            }

            std::vector<std::size_t> counts;
            counts.reserve(dimids.size());
            for (int dimid : dimids) {
                std::size_t length = 0;
                checkNc(nc_inq_dimlen(srcNcid, dimid, &length), "nc_inq_dimlen");
                counts.push_back(length);
            }
            const std::vector<std::size_t> start(dimids.size(), 0);

            if (!usesResizedDimension) {
                copyVariableData(srcNcid, varid, type, start, counts, dstNcid, varid);
                continue;
            }

            if (const auto replacement = resizedDimensionNewData(name)) {
                std::vector<std::size_t> newCounts = counts;
                for (std::size_t i = 0; i < dimids.size(); ++i) {
                    char dimName[NC_MAX_NAME + 1] = {};
                    checkNc(nc_inq_dimname(srcNcid, dimids[i], dimName), "nc_inq_dimname");
                    if (std::string(dimName) == resizedDimensionName) newCounts[i] = resizedDimensionNewLength;
                }
                checkNc(
                    nc_put_vara_float(dstNcid, varid, start.data(), newCounts.data(), replacement->data()), "nc_put_vara_float " + std::string(name));
            }
        }
    } catch (...) {
        nc_close(srcNcid);
        nc_close(dstNcid);
        std::error_code ignored;
        std::filesystem::remove(tmpPath, ignored);
        throw;
    }

    checkNc(nc_close(srcNcid), "nc_close src");
    checkNc(nc_close(dstNcid), "nc_close dst");

    std::error_code error;
    std::filesystem::remove(path, error);
    std::filesystem::rename(tmpPath, path, error);
    if (error) throw UserError("Could not replace " + path.string() + " with rebuilt version: " + error.message());
}

}  // namespace wrftools
