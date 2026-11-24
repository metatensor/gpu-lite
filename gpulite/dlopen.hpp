// gpulite: utilities to find and dynamically load shared libraries, and to
// check the results of calls into them
#ifndef GPULITE_DLOPEN_HPP
#define GPULITE_DLOPEN_HPP

#include <filesystem>
#include <stdexcept>
#include <string>

#if defined(__linux__)
    #include <link.h>
#endif

#if defined(__linux__) || defined(__APPLE__)
    #include <dlfcn.h>
#elif defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif

    #ifndef NOMINMAX
    #define NOMINMAX
    #endif

    #include <windows.h>
    #include <libloaderapi.h>
    #include <psapi.h>

    #include <cassert>

    #include <algorithm>
    #include <optional>
    #include <vector>
#else
    #error "Platform not supported"
#endif

namespace gpulite {
namespace details {

// Define a template to dynamically load symbols
template <typename FuncType> FuncType loadSymbol(void* handle, const char* functionName) {
#if defined(__linux__) || defined(__APPLE__)
    dlerror(); // Clear any existing error
    auto func = reinterpret_cast<FuncType>(dlsym(handle, functionName));

    if (!func) {
        auto* error = dlerror();
        throw std::runtime_error(
            std::string("Failed to load function: ") + functionName + ": " +
            (error ? error : "unknown error")
        );
    }
#elif defined(_WIN32)
    auto func = reinterpret_cast<FuncType>(GetProcAddress(static_cast<HMODULE>(handle), functionName));

    if (!func) {
        auto errorCode = GetLastError();
        LPSTR errorMsg = nullptr;
        FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr,
            errorCode,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            reinterpret_cast<LPSTR>(&errorMsg),
            0,
            nullptr
        );
        auto errorStr = (
            "Failed to load function: " + std::string(functionName) + ": " +
            (errorMsg ? errorMsg : "unknown error")
        );
        LocalFree(errorMsg);
        throw std::runtime_error(std::move(errorStr));
    }
#endif

    return func;
}

#ifdef _WIN32

inline std::wstring GetEnvVar(const wchar_t* name) {
    DWORD n = GetEnvironmentVariableW(name, nullptr, 0);
    if (n == 0) return L"";
    std::wstring val(n, L'\0');
    GetEnvironmentVariableW(name, val.data(), n);
    if (!val.empty() && val.back() == L'\0') val.pop_back();
    return val;
}

// Parse versions from filenames like cudart64_90.dll, cudart64_12.dll, cudart64_120.dll
inline int ParseCudartVersionScore(std::wstring prefix, const std::wstring& filename) {
    // Return a score; higher = preferred. Unknown parse => 0.

    prefix = prefix + L"_";
    // We try to parse digits after `prefix` and before ".dll".
    if (filename.rfind(prefix, 0) != 0) {
        return 0;
    }

    size_t start = prefix.size();
    size_t end = filename.find(L".dll");
    if (end == std::wstring::npos || end <= start) {
        return 0;
    }

    std::wstring num = filename.substr(start, end - start);
    if (num.empty()) {
        return 0;
    }
    for (wchar_t c : num) {
        if (c < L'0' || c > L'9') {
            return 0;
        }
    }

    // e.g. "90" -> 90, "12" -> 12, "120" -> 120
    return std::stoi(num);
}


inline std::vector<std::filesystem::path> CandidateCudaDirs() {
    std::vector<std::filesystem::path> dirs;

    // 1) CUDA_PATH\bin and CUDA_PATH\bin\x64
    std::wstring cudaPath = GetEnvVar(L"CUDA_PATH");
    if (!cudaPath.empty()) {
        dirs.push_back(std::filesystem::path(cudaPath) / L"bin");
        dirs.push_back(std::filesystem::path(cudaPath) / L"bin" / L"x64");
    }

    // 2) Search in PATH
    std::wstring pathEnv = GetEnvVar(L"PATH");
    if (!pathEnv.empty()) {
        size_t start = 0;
        size_t end = pathEnv.find(L';');
        while (end != std::wstring::npos) {
            std::wstring token = pathEnv.substr(start, end - start);
            if (!token.empty()) {
                dirs.push_back(std::filesystem::path(token));
            }
            start = end + 1;
            end = pathEnv.find(L';', start);
        }
        std::wstring token = pathEnv.substr(start);
        if (!token.empty()) {
            dirs.push_back(std::filesystem::path(token));
        }
    }

    // 3) Default toolkit install root (scan versions)
    //    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin
    std::filesystem::path root = L"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA";

    std::error_code ec;
    std::filesystem::directory_options options = std::filesystem::directory_options::skip_permission_denied;
    for (auto& entry: std::filesystem::directory_iterator(root, options, ec)) {
        if (ec) {
            break;
        }

        // folders like v12.4, v13.0, etc.
        if (!entry.is_directory(ec) || ec) {
            continue;
        }

        dirs.push_back(entry.path() / L"bin");
        dirs.push_back(entry.path() / L"bin" / L"x64");
    }

    // De-dup + keep only existing dirs
    std::sort(dirs.begin(), dirs.end());
    dirs.erase(std::unique(dirs.begin(), dirs.end()), dirs.end());
    dirs.erase(
        std::remove_if(dirs.begin(), dirs.end(), [](const std::filesystem::path& p) {
            assert(!p.empty());

            std::error_code ec;
            if (!std::filesystem::exists(p, ec) || ec) {
                return true;
            }

            if (!std::filesystem::is_directory(p, ec) || ec) {
                return true;
            }

            return false;

        }),
        dirs.end()
    );
    return dirs;
}

inline std::optional<std::filesystem::path> FindBestCudaDll(const std::wstring& prefix) {
    auto dirs = CandidateCudaDirs();

    struct Match {
        std::filesystem::path path;
        int score;
    };
    std::vector<Match> matches;

    for (const auto& d: dirs) {
        std::error_code ec;
        std::filesystem::directory_options options = std::filesystem::directory_options::skip_permission_denied;
        for (auto& e: std::filesystem::directory_iterator(d, options, ec)) {
            if (ec) {
                break;
            }

            if (!e.is_regular_file(ec) || ec) {
                continue;
            }

            auto name = e.path().filename().wstring();
            // Must look like <prefix>*.dll
            if (name.size() < prefix.size() + 4) {
                continue;
            }

            if (name.rfind(prefix, 0) != 0) {
                continue;
            }

            if (e.path().extension().wstring() != L".dll") {
                continue;
            }

            int score = ParseCudartVersionScore(prefix, name);
            // Prefer versioned DLLs; still accept plain "<prefix>.dll" with score 1
            if (name == prefix + L".dll") {
                score = std::max(score, 1);
            }

            matches.push_back({ e.path(), score });
        }
    }

    if (matches.empty()) {
        return std::nullopt;
    }

    // Prefer highest score; if tie, prefer shortest path (arbitrary stable tie-break)
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.path.wstring().size() < b.path.wstring().size();
    });

    return matches.front().path;
}

#endif

/// Check the return code of a call into a dynamically loaded library, and
/// throw an exception with the error message if it failed.
template <typename Res, typename ErrFuncType>
auto checkCall(Res status, Res sucess, ErrFuncType errFunc, const char* file, int line, const char* call) {
    if (status != sucess) {
        auto errorString = errFunc(status);
        auto functionCall = std::string(call);
        throw std::runtime_error(
            functionCall + ": failed with error " + errorString +
            " (error code " + std::to_string(status) + ") at " +
            file + ':' + std::to_string(line)
        );
    }
}

inline std::string basename(const std::string& path) {
    auto fs_path = std::filesystem::path(path);
    return fs_path.filename().string();
}

#if defined(_WIN32)
/// Try to find an already loaded library whose name starts with the given
/// prefix, and return its handle if found.
inline HMODULE findLoadedLibrary(const std::wstring& prefix) {
    auto process = GetCurrentProcess();
    DWORD needed = 0;
    auto status = EnumProcessModulesEx(process, nullptr, 0, &needed, LIST_MODULES_ALL);
    if (!status || needed == 0) {
        return nullptr;
    }

    auto modules = std::vector<HMODULE>(needed / sizeof(HMODULE));
    status = EnumProcessModulesEx(
        process,
        modules.data(),
        static_cast<DWORD>(modules.size() * sizeof(HMODULE)),
        &needed,
        LIST_MODULES_ALL
    );
    if (!status) {
        return nullptr;
    }

    std::wstring name(MAX_PATH, L'\0');
    for (auto hmodule : modules) {
        auto n = GetModuleBaseNameW(process, hmodule, name.data(), MAX_PATH);
        if (n == 0) continue;
        name[n] = '\0';
        if (name.find(prefix) == 0) {
            return hmodule;
        }
    }

    return nullptr;
}

#elif defined(__linux__)

inline void* findLoadedLibrary(const char* prefix) {
    struct SearchData {
        const char* prefix;
        void* handle;
    };

    auto data = SearchData{prefix, nullptr};

    auto callback = [](dl_phdr_info* info, std::size_t, void* user_data) -> int {
        auto* data = static_cast<SearchData*>(user_data);
        if (info == nullptr || info->dlpi_name == nullptr || info->dlpi_name[0] == '\0') {
            return 0;
        }

        auto base = basename(info->dlpi_name);
        if (base.find(data->prefix) != 0) {
            return 0;
        }

        void* h = dlopen(info->dlpi_name, RTLD_NOW | RTLD_NOLOAD);
        if (h != nullptr) {
            data->handle = h;
            return 1; // stop iteration
        }

        return 0;
    };

    dl_iterate_phdr(callback, &data);
    return data.handle;
}

#elif defined(__APPLE__)
inline void* findLoadedLibrary(const char* prefix) {
    return nullptr;
}
#endif

} // namespace details
} // namespace gpulite

#endif // GPULITE_DLOPEN_HPP
