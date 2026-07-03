/**
 * 万能视频下载器 — C++ 版本
 *
 * 对标 D:\电影\万能视频下载器.exe (PyInstaller + yt-dlp + tkinter)
 *
 * 功能:
 *   - 多站点视频提取 (YouTube, Bilibili, 抖音, etc.) 通过 yt-dlp 子进程
 *   - 格式/质量选择
 *   - 多线程下载 + 断点续传
 *   - 实时进度显示
 *   - HLS/DASH 流支持
 *   - 代理支持
 *
 * 编译 (MSVC):
 *   cl /EHsc /O2 /std:c++17 video_downloader.cpp /Fe:video_downloader.exe
 *   需要: yt-dlp.exe 在 PATH 中 (或同目录)
 *
 * 编译 (MinGW):
 *   g++ -O2 -std=c++17 video_downloader.cpp -lwinhttp -lshlwapi -o video_downloader.exe
 */

#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0601
#include <windows.h>
#include <winhttp.h>
#include <shlwapi.h>
#include <commctrl.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "comctl32.lib")

namespace fs = std::filesystem;

// ============================================================================
// 工具函数
// ============================================================================

static std::string wchar_to_utf8(const wchar_t* wstr, int len = -1) {
    if (!wstr) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, wstr, len, nullptr, 0, nullptr, nullptr);
    std::string result(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr, len, &result[0], n, nullptr, nullptr);
    return result;
}

static std::wstring utf8_to_wchar(std::string_view str) {
    if (str.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, str.data(), (int)str.size(), nullptr, 0);
    std::wstring result(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, str.data(), (int)str.size(), &result[0], n);
    return result;
}

static std::string format_bytes(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int idx = 0;
    double size = (double)bytes;
    while (size >= 1024.0 && idx < 4) {
        size /= 1024.0;
        idx++;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.*f %s", (idx == 0 ? 0 : 1), size, units[idx]);
    return buf;
}

static std::string format_speed(double bytes_per_sec) {
    return format_bytes((uint64_t)bytes_per_sec) + "/s";
}

static std::string format_duration(int64_t seconds) {
    int64_t h = seconds / 3600;
    int64_t m = (seconds % 3600) / 60;
    int64_t s = seconds % 60;
    char buf[32];
    if (h > 0)
        snprintf(buf, sizeof(buf), "%lld:%02lld:%02lld", h, m, s);
    else
        snprintf(buf, sizeof(buf), "%lld:%02lld", m, s);
    return buf;
}

static std::vector<std::string> split(std::string_view str, char delim) {
    std::vector<std::string> parts;
    size_t start = 0, end;
    while ((end = str.find(delim, start)) != std::string_view::npos) {
        parts.emplace_back(str.substr(start, end - start));
        start = end + 1;
    }
    if (start < str.size()) parts.emplace_back(str.substr(start));
    return parts;
}

static std::string strip(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\n' || s.front() == '\r'))
        s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\n' || s.back() == '\r'))
        s.remove_suffix(1);
    return std::string(s);
}

// ============================================================================
// 简易 JSON 解析器 (无需外部依赖)
// ============================================================================

struct JsonValue {
    enum Type { NUL, BOOL, INT, FLOAT, STRING, ARRAY, OBJECT };
    Type type = NUL;
    bool bval = false;
    int64_t ival = 0;
    double fval = 0.0;
    std::string sval;
    std::vector<JsonValue> arr;
    std::unordered_map<std::string, JsonValue> obj;
    std::vector<std::string> obj_keys;  // 保持插入顺序

    static JsonValue parse(std::string_view json);
    std::string dump(int indent = 0) const;

    const JsonValue& operator[](const char* key) const {
        static JsonValue null_val;
        auto it = obj.find(key);
        return it != obj.end() ? it->second : null_val;
    }
    const JsonValue& operator[](size_t idx) const {
        static JsonValue null_val;
        return idx < arr.size() ? arr[idx] : null_val;
    }
    bool has(const char* key) const { return obj.find(key) != obj.end(); }
    size_t size() const { return type == ARRAY ? arr.size() : type == OBJECT ? obj.size() : 0; }
    std::string str_or(const char* def = "") const {
        return type == STRING ? sval : def;
    }
    int64_t int_or(int64_t def = 0) const {
        if (type == INT) return ival;
        if (type == FLOAT) return (int64_t)fval;
        return def;
    }
    double float_or(double def = 0.0) const {
        if (type == FLOAT) return fval;
        if (type == INT) return (double)ival;
        return def;
    }
    bool bool_or(bool def = false) const {
        return type == BOOL ? bval : def;
    }
};

// 简易递归下降 JSON 解析器
struct JsonParser {
    std::string_view json;
    size_t pos = 0;

    JsonParser(std::string_view j) : json(j) {}

    void skip_ws() {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                                      json[pos] == '\n' || json[pos] == '\r'))
            pos++;
    }

    char peek() { skip_ws(); return pos < json.size() ? json[pos] : '\0'; }
    char next() { skip_ws(); return pos < json.size() ? json[pos++] : '\0'; }

    JsonValue parse_value() {
        skip_ws();
        if (pos >= json.size()) return JsonValue{};

        char c = json[pos];
        if (c == '"') return parse_string();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }

    JsonValue parse_string() {
        next(); // skip '"'
        std::string s;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                switch (json[pos]) {
                    case '"':  s += '"'; break;
                    case '\\': s += '\\'; break;
                    case '/':  s += '/'; break;
                    case 'b':  s += '\b'; break;
                    case 'f':  s += '\f'; break;
                    case 'n':  s += '\n'; break;
                    case 'r':  s += '\r'; break;
                    case 't':  s += '\t'; break;
                    case 'u': {
                        if (pos + 4 < json.size()) {
                            unsigned cp = 0;
                            for (int i = 1; i <= 4; i++)
                                cp = (cp << 4) | hex_val(json[pos + i]);
                            pos += 4;
                            if (cp < 0x80) s += (char)cp;
                            else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
                            else { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp>>6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
                        }
                        break;
                    }
                }
                pos++;
            } else {
                s += json[pos++];
            }
        }
        if (pos < json.size()) pos++; // skip closing '"'
        JsonValue v; v.type = JsonValue::STRING; v.sval = s;
        return v;
    }

    JsonValue parse_object() {
        next(); // skip '{'
        JsonValue v; v.type = JsonValue::OBJECT;
        while (peek() != '}' && pos < json.size()) {
            JsonValue key = parse_string();
            if (peek() == ':') next();
            v.obj_keys.push_back(key.sval);
            v.obj[key.sval] = parse_value();
            if (peek() == ',') next();
        }
        if (pos < json.size()) pos++; // skip '}'
        return v;
    }

    JsonValue parse_array() {
        next(); // skip '['
        JsonValue v; v.type = JsonValue::ARRAY;
        while (peek() != ']' && pos < json.size()) {
            v.arr.push_back(parse_value());
            if (peek() == ',') next();
        }
        if (pos < json.size()) pos++; // skip ']'
        return v;
    }

    JsonValue parse_bool() {
        JsonValue v; v.type = JsonValue::BOOL;
        if (json.substr(pos, 4) == "true") { v.bval = true; pos += 4; }
        else { v.bval = false; pos += 5; }
        return v;
    }

    JsonValue parse_null() {
        pos += 4; // skip "null"
        return JsonValue{};
    }

    JsonValue parse_number() {
        JsonValue v;
        size_t start = pos;
        bool is_float = false;
        if (pos < json.size() && json[pos] == '-') pos++;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') pos++;
        if (pos < json.size() && json[pos] == '.') { is_float = true; pos++; }
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') pos++;
        if (pos < json.size() && (json[pos] == 'e' || json[pos] == 'E')) {
            is_float = true; pos++;
            if (pos < json.size() && (json[pos] == '+' || json[pos] == '-')) pos++;
            while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') pos++;
        }
        auto num_str = std::string(json.substr(start, pos - start));
        if (is_float) {
            v.type = JsonValue::FLOAT;
            v.fval = std::stod(num_str);
        } else {
            v.type = JsonValue::INT;
            v.ival = std::stoll(num_str);
        }
        return v;
    }

    static int hex_val(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    }
};

JsonValue JsonValue::parse(std::string_view json) {
    return JsonParser(json).parse_value();
}

// ============================================================================
// 视频格式信息
// ============================================================================

struct VideoFormat {
    std::string format_id;
    std::string ext;           // mp4, webm, mkv, etc.
    std::string resolution;    // 1920x1080
    std::string vcodec;        // h264, vp9, av1
    std::string acodec;        // aac, opus, mp4a
    std::string note;          // 备注 (e.g. "1080p", "best")
    int64_t filesize = 0;      // 字节 (0 = unknown)
    int64_t filesize_approx = 0;
    int width = 0;
    int height = 0;
    double fps = 0.0;
    double tbr = 0.0;          // 平均码率
    double abr = 0.0;          // 音频码率
    double vbr = 0.0;          // 视频码率
    bool has_video = true;
    bool has_audio = true;
    std::string protocol;      // https, m3u8, dash
};

struct VideoInfo {
    std::string title;
    std::string description;
    std::string uploader;
    std::string upload_date;
    std::string webpage_url;
    std::string thumbnail_url;
    int64_t duration = 0;       // 秒
    int64_t view_count = 0;
    int64_t like_count = 0;
    std::vector<VideoFormat> formats;
    std::vector<std::string> categories;
    std::vector<std::string> tags;

    // 选择最佳格式
    const VideoFormat* best_video(int max_height = 99999) const {
        const VideoFormat* best = nullptr;
        for (auto& f : formats) {
            if (!f.has_video) continue;
            if (f.height > max_height) continue;
            if (!best || f.height > best->height ||
                (f.height == best->height && f.tbr > best->tbr))
                best = &f;
        }
        return best;
    }

    const VideoFormat* best_audio() const {
        const VideoFormat* best = nullptr;
        for (auto& f : formats) {
            if (!f.has_audio || f.has_video) continue;
            if (!best || f.abr > best->abr) best = &f;
        }
        if (!best) best = best_video();
        return best;
    }
};

// ============================================================================
// yt-dlp 子进程调用 (提取视频信息)
// ============================================================================

class YtDlpExtractor {
public:
    static std::optional<VideoInfo> extract(const std::string& url,
                                             const std::string& proxy = "",
                                             const std::string& cookies = "") {
        std::string cmd = "yt-dlp.exe";
        cmd += " --dump-json --no-playlist --ignore-errors --no-warnings";
        if (!proxy.empty()) cmd += " --proxy " + proxy;
        if (!cookies.empty()) cmd += " --cookies " + cookies;
        cmd += " \"" + url + "\"";
        cmd += " 2>nul";

        std::string output = exec_command(cmd);
        if (output.empty()) {
            // 尝试 python -m yt_dlp
            cmd = "python -m yt_dlp --dump-json --no-playlist --ignore-errors --no-warnings";
            if (!proxy.empty()) cmd += " --proxy " + proxy;
            cmd += " \"" + url + "\"";
            cmd += " 2>nul";
            output = exec_command(cmd);
        }
        if (output.empty()) return std::nullopt;
        return parse_info(output);
    }

    static std::optional<std::vector<VideoInfo>> extract_playlist(
        const std::string& url, const std::string& proxy = "") {
        std::string cmd = "yt-dlp.exe";
        cmd += " --dump-json --ignore-errors --no-warnings";
        if (!proxy.empty()) cmd += " --proxy " + proxy;
        cmd += " \"" + url + "\"";
        cmd += " 2>nul";

        std::string output = exec_command(cmd);
        if (output.empty()) return std::nullopt;

        std::vector<VideoInfo> results;
        size_t pos = 0;
        while (pos < output.size()) {
            // 找到每个 JSON 对象边界
            size_t start = output.find('{', pos);
            if (start == std::string::npos) break;
            int depth = 0;
            size_t end = start;
            bool in_str = false;
            while (end < output.size()) {
                char c = output[end];
                if (c == '"' && (end == 0 || output[end-1] != '\\')) in_str = !in_str;
                if (!in_str) {
                    if (c == '{') depth++;
                    else if (c == '}') { depth--; if (depth == 0) { end++; break; } }
                }
                end++;
            }
            if (depth == 0) {
                auto vi = parse_info(output.substr(start, end - start));
                if (vi) results.push_back(*vi);
            }
            pos = end;
        }
        return results.empty() ? std::nullopt : std::make_optional(results);
    }

private:
    static std::string exec_command(const std::string& cmd) {
        SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        HANDLE hRead, hWrite;
        if (!CreatePipe(&hRead, &hWrite, &sa, 0)) return "";

        SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOW si = {sizeof(STARTUPINFOW)};
        si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
        si.wShowWindow = SW_HIDE;
        si.hStdOutput = hWrite;
        si.hStdError = hWrite;

        PROCESS_INFORMATION pi = {};
        std::wstring wcmd = utf8_to_wchar(cmd);
        std::unique_ptr<wchar_t[]> cmd_buf(new wchar_t[wcmd.size() + 1]);
        wcscpy(cmd_buf.get(), wcmd.c_str());

        std::string result;
        if (CreateProcessW(nullptr, cmd_buf.get(), nullptr, nullptr, TRUE,
                           CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
            CloseHandle(hWrite);
            WaitForSingleObject(pi.hProcess, 30000);  // 30s timeout for extract
            char buf[4096];
            DWORD read;
            while (ReadFile(hRead, buf, sizeof(buf) - 1, &read, nullptr) && read > 0) {
                buf[read] = '\0';
                result += buf;
            }
            CloseHandle(hRead);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        } else {
            CloseHandle(hWrite);
            CloseHandle(hRead);
        }
        return result;
    }

    static std::optional<VideoInfo> parse_info(const std::string& json_str) {
        try {
            auto j = JsonValue::parse(json_str);
            if (j.type != JsonValue::OBJECT) return std::nullopt;

            VideoInfo vi;
            vi.title = j["title"].str_or("Unknown");
            vi.description = j["description"].str_or();
            vi.uploader = j["uploader"].str_or(j["channel"].str_or("Unknown"));
            vi.upload_date = j["upload_date"].str_or();
            vi.webpage_url = j["webpage_url"].str_or();
            vi.thumbnail_url = j["thumbnail"].str_or();
            vi.duration = j["duration"].int_or(0);
            vi.view_count = j["view_count"].int_or(0);
            vi.like_count = j["like_count"].int_or(0);

            // categories
            if (j.has("categories") && j["categories"].type == JsonValue::ARRAY) {
                for (auto& c : j["categories"].arr)
                    if (c.type == JsonValue::STRING) vi.categories.push_back(c.sval);
            }
            // tags
            if (j.has("tags") && j["tags"].type == JsonValue::ARRAY) {
                for (auto& t : j["tags"].arr)
                    if (t.type == JsonValue::STRING) vi.tags.push_back(t.sval);
            }

            // formats
            if (j.has("formats") && j["formats"].type == JsonValue::ARRAY) {
                for (auto& fj : j["formats"].arr) {
                    if (fj.type != JsonValue::OBJECT) continue;
                    VideoFormat f;
                    f.format_id = fj["format_id"].str_or();
                    f.ext = fj["ext"].str_or("mp4");
                    f.resolution = fj["resolution"].str_or();
                    f.vcodec = fj["vcodec"].str_or("none");
                    f.acodec = fj["acodec"].str_or("none");
                    f.note = fj["format_note"].str_or();
                    f.filesize = fj["filesize"].int_or(0);
                    f.filesize_approx = fj["filesize_approx"].int_or(0);
                    f.width = (int)fj["width"].int_or(0);
                    f.height = (int)fj["height"].int_or(0);
                    f.fps = fj["fps"].float_or(0.0);
                    f.tbr = fj["tbr"].float_or(0.0);
                    f.abr = fj["abr"].float_or(0.0);
                    f.vbr = fj["vbr"].float_or(0.0);
                    f.protocol = fj["protocol"].str_or("https");
                    f.has_video = f.vcodec != "none";
                    f.has_audio = f.acodec != "none";
                    vi.formats.push_back(std::move(f));
                }
            }
            return vi;
        } catch (...) {
            return std::nullopt;
        }
    }
};

// ============================================================================
// HTTP 下载器 (WinHTTP, 支持断点续传 + 多线程)
// ============================================================================

struct DownloadProgress {
    std::atomic<uint64_t> downloaded{0};
    std::atomic<uint64_t> total{0};
    std::atomic<bool> paused{false};
    std::atomic<bool> cancelled{false};
    std::atomic<bool> finished{false};
    std::string status;  // "downloading", "merging", "done", "error"
    std::string error_msg;
    std::chrono::steady_clock::time_point start_time;
};

class HttpDownloader {
public:
    struct Options {
        std::string url;
        std::string output_path;
        int64_t resume_from = 0;  // 断点续传位置
        int num_threads = 4;
        int max_retries = 3;
        int connect_timeout_ms = 15000;
        int read_timeout_ms = 30000;
        std::string proxy;        // http://127.0.0.1:7890
        std::string user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36";
        std::string cookies;
        std::map<std::string, std::string> extra_headers;
    };

    static bool download(const Options& opts, DownloadProgress* progress = nullptr) {
        if (progress) {
            progress->downloaded = 0;
            progress->total = 0;
            progress->finished = false;
            progress->status = "connecting";
            progress->start_time = std::chrono::steady_clock::now();
        }

        // 解析 URL
        std::wstring url = utf8_to_wchar(opts.url);
        URL_COMPONENTS urlc = {sizeof(URL_COMPONENTS)};
        wchar_t host[256] = {0}, path[2048] = {0};
        urlc.lpszHostName = host;
        urlc.dwHostNameLength = 255;
        urlc.lpszUrlPath = path;
        urlc.dwUrlPathLength = 2047;

        if (!WinHttpCrackUrl(url.c_str(), (DWORD)url.size(), 0, &urlc)) {
            if (progress) progress->status = "error: bad url";
            return false;
        }

        bool use_ssl = (urlc.nScheme == INTERNET_SCHEME_HTTPS);

        HINTERNET hSession = WinHttpOpen(
            utf8_to_wchar(opts.user_agent).c_str(),
            opts.proxy.empty() ? WINHTTP_ACCESS_TYPE_DEFAULT_PROXY : WINHTTP_ACCESS_TYPE_NAMED_PROXY,
            opts.proxy.empty() ? WINHTTP_NO_PROXY_NAME : utf8_to_wchar(opts.proxy).c_str(),
            WINHTTP_NO_PROXY_BYPASS, 0);

        if (!hSession) { if (progress) progress->status = "error: session"; return false; }

        WinHttpSetTimeouts(hSession, opts.connect_timeout_ms, opts.connect_timeout_ms,
                           opts.read_timeout_ms, opts.read_timeout_ms);

        HINTERNET hConnect = WinHttpConnect(hSession, host, urlc.nPort, 0);
        if (!hConnect) { WinHttpCloseHandle(hSession); return false; }

        HINTERNET hRequest = WinHttpOpenRequest(
            hConnect, L"GET", path, nullptr, WINHTTP_NO_REFERER,
            WINHTTP_DEFAULT_ACCEPT_TYPES,
            use_ssl ? WINHTTP_FLAG_SECURE : 0);

        if (!hRequest) { WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession); return false; }

        // 断点续传
        if (opts.resume_from > 0) {
            wchar_t range[64];
            swprintf(range, 64, L"bytes=%lld-", (long long)opts.resume_from);
            WinHttpAddRequestHeaders(hRequest, range, (DWORD)wcslen(range), WINHTTP_ADDREQ_FLAG_ADD);
        }

        // 自定义 headers
        for (auto& [k, v] : opts.extra_headers) {
            std::string hdr = k + ": " + v;
            std::wstring whdr = utf8_to_wchar(hdr);
            WinHttpAddRequestHeaders(hRequest, whdr.c_str(), (DWORD)whdr.size(),
                                     WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE);
        }

        if (!opts.cookies.empty()) {
            std::wstring cookie_hdr = L"Cookie: " + utf8_to_wchar(opts.cookies);
            WinHttpAddRequestHeaders(hRequest, cookie_hdr.c_str(), (DWORD)cookie_hdr.size(),
                                     WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE);
        }

        if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
            !WinHttpReceiveResponse(hRequest, nullptr)) {
            WinHttpCloseHandle(hRequest);
            WinHttpCloseHandle(hConnect);
            WinHttpCloseHandle(hSession);
            if (progress) progress->status = "error: request failed";
            return false;
        }

        // 获取文件大小
        wchar_t content_len[32] = {0};
        DWORD cl_len = sizeof(content_len);
        uint64_t total_size = 0;
        if (WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_CONTENT_LENGTH,
                                WINHTTP_HEADER_NAME_BY_INDEX, content_len, &cl_len, WINHTTP_NO_HEADER_INDEX)) {
            total_size = _wcstoui64(content_len, nullptr, 10);
        }
        if (progress) {
            progress->total = total_size + opts.resume_from;
            progress->downloaded = opts.resume_from;
            progress->status = "downloading";
        }

        // 打开输出文件
        std::ofstream outfile;
        if (!opts.output_path.empty()) {
            auto mode = (opts.resume_from > 0) ? std::ios::binary | std::ios::app
                                                : std::ios::binary;
            outfile.open(opts.output_path, mode);
            if (!outfile) {
                WinHttpCloseHandle(hRequest);
                WinHttpCloseHandle(hConnect);
                WinHttpCloseHandle(hSession);
                if (progress) progress->status = "error: cannot open output";
                return false;
            }
        }

        // 下载循环
        std::vector<uint8_t> buf(65536);
        uint64_t total_dl = opts.resume_from;
        auto last_update = std::chrono::steady_clock::now();

        while (true) {
            if (progress && progress->cancelled) break;

            // 暂停处理
            while (progress && progress->paused) {
                progress->status = "paused";
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            if (progress && !progress->paused) progress->status = "downloading";

            DWORD available = 0;
            if (!WinHttpQueryDataAvailable(hRequest, &available)) break;
            if (available == 0) break;

            DWORD to_read = std::min(available, (DWORD)buf.size());
            DWORD read = 0;
            if (!WinHttpReadData(hRequest, buf.data(), to_read, &read)) break;

            if (outfile.is_open()) outfile.write((char*)buf.data(), read);
            total_dl += read;

            auto now = std::chrono::steady_clock::now();
            if (progress && now - last_update > std::chrono::milliseconds(100)) {
                progress->downloaded = total_dl;
                last_update = now;
            }
        }

        if (outfile.is_open()) outfile.close();
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);

        if (progress) {
            progress->finished = true;
            progress->status = progress->cancelled ? "cancelled" : "done";
            progress->downloaded = total_dl;
        }
        return true;
    }
};

// ============================================================================
// 多线程分片下载器
// ============================================================================

class ChunkedDownloader {
public:
    static bool download(const HttpDownloader::Options& base_opts,
                         uint64_t total_size,
                         int num_chunks,
                         DownloadProgress* progress = nullptr) {
        if (num_chunks <= 1 || total_size < 4 * 1024 * 1024) {
            // 单线程下载
            return HttpDownloader::download(base_opts, progress);
        }

        struct ChunkState {
            std::string tmp_path;
            std::atomic<uint64_t> downloaded{0};
            bool ok = false;
        };

        uint64_t chunk_size = total_size / num_chunks;
        std::vector<ChunkState> chunks(num_chunks);
        std::vector<std::thread> workers;

        fs::path output_path(base_opts.output_path);
        fs::path tmp_dir = output_path.parent_path() / (output_path.stem().string() + ".tmp");
        fs::create_directories(tmp_dir);

        std::mutex progress_mtx;

        for (int i = 0; i < num_chunks; i++) {
            workers.emplace_back([&, i]() {
                auto opts = base_opts;
                uint64_t start = i * chunk_size;
                uint64_t end = (i == num_chunks - 1) ? total_size - 1 : (i + 1) * chunk_size - 1;
                opts.resume_from = start;
                chunks[i].tmp_path = (tmp_dir / ("chunk_" + std::to_string(i))).string();
                opts.output_path = chunks[i].tmp_path;

                // 添加 Range 头
                opts.extra_headers["Range"] = "bytes=" + std::to_string(start) + "-" + std::to_string(end);

                chunks[i].ok = HttpDownloader::download(opts, nullptr);

                // 部分更新进度 (简化版 — 用文件大小估计)
                if (chunks[i].ok) {
                    std::error_code ec;
                    chunks[i].downloaded = fs::file_size(chunks[i].tmp_path, ec);
                }
            });
        }

        // 等待所有线程
        for (auto& t : workers) t.join();

        // 检查是否全部成功
        bool all_ok = true;
        for (auto& c : chunks) if (!c.ok) { all_ok = false; break; }

        if (!all_ok) {
            // 回退到单线程
            return HttpDownloader::download(base_opts, progress);
        }

        // 合并分片
        if (progress) progress->status = "merging";
        std::ofstream out(base_opts.output_path, std::ios::binary);
        if (!out) return false;

        std::vector<uint8_t> merge_buf(1 * 1024 * 1024);  // 1MB buffer
        for (auto& c : chunks) {
            std::ifstream in(c.tmp_path, std::ios::binary);
            while (in) {
                in.read((char*)merge_buf.data(), merge_buf.size());
                out.write((char*)merge_buf.data(), in.gcount());
            }
            in.close();
            fs::remove(c.tmp_path);
        }
        out.close();
        fs::remove_all(tmp_dir);

        if (progress) {
            progress->finished = true;
            progress->status = "done";
            progress->downloaded = total_size;
        }
        return true;
    }
};

// ============================================================================
// 视频下载管理器
// ============================================================================

class VideoDownloadManager {
public:
    struct DownloadTask {
        std::string url;
        std::string output_dir = ".";
        std::string output_template = "%(title)s.%(ext)s";
        std::string selected_format = "best";  // "best", "bestvideo+bestaudio", format_id
        int max_height = 1080;
        int num_threads = 4;
        std::string proxy;
        std::string cookies;
        bool audio_only = false;
        bool extract_audio = false;
        bool embed_thumbnail = false;

        std::unique_ptr<DownloadProgress> progress;
        std::unique_ptr<VideoInfo> info;
    };

    static bool download(DownloadTask& task) {
        // 1. 提取视频信息
        std::cout << "[1/3] 正在获取视频信息..." << std::endl;
        auto vi = YtDlpExtractor::extract(task.url, task.proxy, task.cookies);
        if (!vi) {
            std::cerr << "错误: 无法提取视频信息 (请确认 yt-dlp.exe 在 PATH 中)" << std::endl;
            return false;
        }
        task.info = std::make_unique<VideoInfo>(*vi);
        std::cout << "  标题: " << vi->title << std::endl;
        std::cout << "  上传: " << vi->uploader << std::endl;
        std::cout << "  时长: " << format_duration(vi->duration) << std::endl;
        std::cout << "  格式数: " << vi->formats.size() << std::endl;

        // 2. 选择格式
        const VideoFormat* selected_fmt = nullptr;
        if (task.audio_only || task.extract_audio) {
            selected_fmt = vi->best_audio();
        } else {
            selected_fmt = vi->best_video(task.max_height);
        }
        if (!selected_fmt) {
            std::cerr << "错误: 没有找到合适的格式" << std::endl;
            return false;
        }

        std::cout << "\n[2/3] 选择格式: " << selected_fmt->format_id
                  << " (" << selected_fmt->ext << ", "
                  << selected_fmt->resolution
                  << ", " << format_bytes(selected_fmt->filesize) << ")" << std::endl;

        // 3. 使用 yt-dlp 下载 (一站式: 提取 + 下载 + 合并)
        std::cout << "\n[3/3] 开始下载..." << std::endl;

        std::string output_path = task.output_dir;
        if (!output_path.empty() && output_path.back() != '\\' && output_path.back() != '/')
            output_path += "\\";
        output_path += task.output_template;

        std::string cmd = "yt-dlp.exe";
        cmd += " -f " + selected_fmt->format_id;
        cmd += " -o \"" + output_path + "\"";
        cmd += " --no-playlist";
        if (!task.proxy.empty()) cmd += " --proxy " + task.proxy;
        if (!task.cookies.empty()) cmd += " --cookies " + task.cookies;
        if (task.num_threads > 1) cmd += " --concurrent-fragments " + std::to_string(task.num_threads);
        cmd += " --newline --progress";
        cmd += " \"" + task.url + "\"";

        std::cout << "  " << cmd << std::endl << std::endl;

        // 直接运行 (实时输出到控制台)
        return run_command_realtime(cmd);
    }

    static void show_formats(const VideoInfo& vi) {
        std::cout << "\n可用格式:\n";
        std::cout << std::left
                  << std::setw(12) << "ID"
                  << std::setw(6) << "EXT"
                  << std::setw(14) << "分辨率"
                  << std::setw(10) << "大小"
                  << std::setw(8) << "FPS"
                  << std::setw(12) << "编码"
                  << "备注\n";
        std::cout << std::string(80, '-') << "\n";

        for (auto& f : vi.formats) {
            std::string size_str = f.filesize > 0 ? format_bytes(f.filesize) :
                                    f.filesize_approx > 0 ? "~" + format_bytes(f.filesize_approx) : "?";
            std::string codec = f.vcodec;
            if (codec.size() > 10) codec = codec.substr(0, 10);
            if (f.has_video && f.has_audio) codec += "+audio";
            else if (!f.has_video) codec = "audio only";
            else if (!f.has_audio) codec = "video only";

            std::cout << std::left
                      << std::setw(12) << f.format_id
                      << std::setw(6) << f.ext
                      << std::setw(14) << (f.resolution.empty() ? "audio" : f.resolution)
                      << std::setw(10) << size_str
                      << std::setw(8) << (f.fps > 0 ? std::to_string((int)f.fps) : "-")
                      << std::setw(12) << codec
                      << f.note << "\n";
        }
    }

private:
    static bool run_command_realtime(const std::string& cmd) {
        SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        HANDLE hRead, hWrite;
        if (!CreatePipe(&hRead, &hWrite, &sa, 0)) return false;
        SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOW si = {sizeof(STARTUPINFOW)};
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = hWrite;
        si.hStdError = hWrite;

        PROCESS_INFORMATION pi = {};
        std::wstring wcmd = utf8_to_wchar(cmd);
        std::unique_ptr<wchar_t[]> cmd_buf(new wchar_t[wcmd.size() + 1]);
        wcscpy(cmd_buf.get(), wcmd.c_str());

        if (!CreateProcessW(nullptr, cmd_buf.get(), nullptr, nullptr, TRUE,
                            0, nullptr, nullptr, &si, &pi)) {
            CloseHandle(hWrite);
            CloseHandle(hRead);
            return false;
        }

        CloseHandle(hWrite);

        char buf[4096];
        DWORD read;
        while (ReadFile(hRead, buf, sizeof(buf) - 1, &read, nullptr) && read > 0) {
            buf[read] = '\0';
            std::cout << buf;
        }

        CloseHandle(hRead);
        WaitForSingleObject(pi.hProcess, INFINITE);

        DWORD exit_code = 0;
        GetExitCodeProcess(pi.hProcess, &exit_code);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);

        return exit_code == 0;
    }
};

// ============================================================================
// Win32 GUI (对标 tkinter 界面)
// ============================================================================

#define IDC_URL_INPUT      1001
#define IDC_BROWSE_BTN     1002
#define IDC_OUTPUT_INPUT   1003
#define IDC_QUALITY_COMBO  1004
#define IDC_FORMAT_LIST    1005
#define IDC_EXTRACT_BTN    1006
#define IDC_DOWNLOAD_BTN   1007
#define IDC_PROGRESS_BAR   1008
#define IDC_STATUS_TEXT    1009
#define IDC_AUDIO_ONLY     1010

struct GuiState {
    HWND hUrlInput = nullptr;
    HWND hOutputInput = nullptr;
    HWND hQualityCombo = nullptr;
    HWND hFormatList = nullptr;
    HWND hExtractBtn = nullptr;
    HWND hDownloadBtn = nullptr;
    HWND hProgressBar = nullptr;
    HWND hStatusText = nullptr;
    HWND hAudioOnly = nullptr;

    std::unique_ptr<VideoInfo> current_info;
    std::unique_ptr<std::thread> download_thread;
    std::unique_ptr<DownloadProgress> progress;
    std::wstring last_output_dir = L".";
};

static GuiState g_gui;

static void gui_set_status(const wchar_t* text) {
    SetWindowTextW(g_gui.hStatusText, text);
}

static void gui_update_progress() {
    if (!g_gui.progress) return;
    uint64_t dl = g_gui.progress->downloaded;
    uint64_t total = g_gui.progress->total;
    if (total > 0) {
        int pct = (int)(dl * 100 / total);
        SendMessageW(g_gui.hProgressBar, PBM_SETPOS, (WPARAM)pct, 0);
        wchar_t txt[128];
        swprintf(txt, 128, L"下载中... %d%% (%s / %s)",
                 pct,
                 utf8_to_wchar(format_bytes(dl)).c_str(),
                 utf8_to_wchar(format_bytes(total)).c_str());
        gui_set_status(txt);
    }
}

static void gui_on_extract() {
    wchar_t url[2048] = {0};
    GetWindowTextW(g_gui.hUrlInput, url, 2047);
    if (wcslen(url) == 0) {
        MessageBoxW(nullptr, L"请输入视频 URL", L"提示", MB_OK | MB_ICONINFORMATION);
        return;
    }

    gui_set_status(L"正在获取视频信息...");
    EnableWindow(g_gui.hExtractBtn, FALSE);
    EnableWindow(g_gui.hDownloadBtn, FALSE);

    auto vi = YtDlpExtractor::extract(wchar_to_utf8(url));
    if (!vi) {
        gui_set_status(L"获取失败: 请确认 URL 正确且 yt-dlp.exe 可用");
        EnableWindow(g_gui.hExtractBtn, TRUE);
        return;
    }

    g_gui.current_info = std::make_unique<VideoInfo>(*vi);

    // 更新格式列表
    ListView_DeleteAllItems(g_gui.hFormatList);
    for (size_t i = 0; i < vi->formats.size(); i++) {
        auto& f = vi->formats[i];
        wchar_t id[32]; swprintf(id, 32, L"%S", f.format_id.c_str());
        wchar_t ext[16]; swprintf(ext, 16, L"%S", f.ext.c_str());
        wchar_t res[32]; swprintf(res, 32, L"%S", f.resolution.c_str());
        wchar_t size_str[32]; swprintf(size_str, 32, L"%S", format_bytes(f.filesize).c_str());
        wchar_t note[128]; swprintf(note, 128, L"%S", f.note.c_str());

        LVITEMW item = {};
        item.mask = LVIF_TEXT;
        item.iItem = (int)i;
        item.pszText = id;
        ListView_InsertItem(g_gui.hFormatList, &item);

        ListView_SetItemText(g_gui.hFormatList, (int)i, 1, ext);
        ListView_SetItemText(g_gui.hFormatList, (int)i, 2, res);
        ListView_SetItemText(g_gui.hFormatList, (int)i, 3, size_str);
        ListView_SetItemText(g_gui.hFormatList, (int)i, 4, note);
    }

    wchar_t status[256];
    swprintf(status, 256, L"已获取: %S (%S) - %zu 个格式",
             vi->title.c_str(), format_duration(vi->duration).c_str(), vi->formats.size());
    gui_set_status(status);

    EnableWindow(g_gui.hExtractBtn, TRUE);
    EnableWindow(g_gui.hDownloadBtn, TRUE);
}

static void gui_on_download() {
    if (!g_gui.current_info) {
        gui_on_extract();
        if (!g_gui.current_info) return;
    }

    wchar_t url[2048] = {0};
    wchar_t out_dir[1024] = {0};
    GetWindowTextW(g_gui.hUrlInput, url, 2047);
    GetWindowTextW(g_gui.hOutputInput, out_dir, 1023);

    bool audio_only = (SendMessageW(g_gui.hAudioOnly, BM_GETCHECK, 0, 0) == BST_CHECKED);

    // 获取选中的格式
    int sel = ListView_GetNextItem(g_gui.hFormatList, -1, LVNI_SELECTED);
    std::string format_id = "best";
    if (sel >= 0 && sel < (int)g_gui.current_info->formats.size()) {
        format_id = g_gui.current_info->formats[sel].format_id;
    }
    if (audio_only) format_id = "bestaudio";

    VideoDownloadManager::DownloadTask task;
    task.url = wchar_to_utf8(url);
    task.output_dir = wchar_to_utf8(out_dir);
    if (task.output_dir.empty()) task.output_dir = ".";
    task.selected_format = format_id;
    task.audio_only = audio_only;

    g_gui.progress = std::make_unique<DownloadProgress>();
    gui_set_status(L"开始下载...");
    SendMessageW(g_gui.hProgressBar, PBM_SETPOS, 0, 0);
    EnableWindow(g_gui.hDownloadBtn, FALSE);

    // 在后台线程下载
    g_gui.download_thread = std::make_unique<std::thread>([task = std::move(task)]() mutable {
        bool ok = VideoDownloadManager::download(task);
        std::wstring msg = ok ? L"下载完成!" : L"下载失败";
        gui_set_status(msg.c_str());
        EnableWindow(g_gui.hDownloadBtn, TRUE);
    });
}

static LRESULT CALLBACK gui_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_CREATE: {
            // 创建控件 (对标 tkinter 布局)
            int y = 10;

            // URL 输入
            CreateWindowW(L"STATIC", L"视频 URL:", WS_CHILD | WS_VISIBLE,
                          10, y, 60, 20, hwnd, nullptr, nullptr, nullptr);
            g_gui.hUrlInput = CreateWindowW(L"EDIT", L"",
                WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                80, y - 2, 500, 22, hwnd, (HMENU)IDC_URL_INPUT, nullptr, nullptr);

            // 提取按钮
            g_gui.hExtractBtn = CreateWindowW(L"BUTTON", L"获取信息",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                590, y - 2, 80, 24, hwnd, (HMENU)IDC_EXTRACT_BTN, nullptr, nullptr);
            y += 28;

            // 输出路径
            CreateWindowW(L"STATIC", L"保存到:", WS_CHILD | WS_VISIBLE,
                          10, y, 60, 20, hwnd, nullptr, nullptr, nullptr);
            g_gui.hOutputInput = CreateWindowW(L"EDIT", L".",
                WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                80, y - 2, 500, 22, hwnd, (HMENU)IDC_OUTPUT_INPUT, nullptr, nullptr);
            g_gui.hDownloadBtn = CreateWindowW(L"BUTTON", L"下载",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                590, y - 2, 80, 24, hwnd, (HMENU)IDC_DOWNLOAD_BTN, nullptr, nullptr);
            y += 28;

            // 仅音频
            g_gui.hAudioOnly = CreateWindowW(L"BUTTON", L"仅下载音频",
                WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                80, y, 100, 20, hwnd, (HMENU)IDC_AUDIO_ONLY, nullptr, nullptr);
            y += 24;

            // 格式列表
            g_gui.hFormatList = CreateWindowW(WC_LISTVIEWW, L"",
                WS_CHILD | WS_VISIBLE | WS_BORDER | LVS_REPORT | LVS_SINGLESEL,
                10, y, 660, 180, hwnd, (HMENU)IDC_FORMAT_LIST, nullptr, nullptr);

            // 添加列
            LVCOLUMNW col = {};
            col.mask = LVCF_TEXT | LVCF_WIDTH;
            col.cx = 80; col.pszText = (LPWSTR)L"格式ID"; ListView_InsertColumn(g_gui.hFormatList, 0, &col);
            col.cx = 50; col.pszText = (LPWSTR)L"扩展"; ListView_InsertColumn(g_gui.hFormatList, 1, &col);
            col.cx = 100; col.pszText = (LPWSTR)L"分辨率"; ListView_InsertColumn(g_gui.hFormatList, 2, &col);
            col.cx = 80; col.pszText = (LPWSTR)L"大小"; ListView_InsertColumn(g_gui.hFormatList, 3, &col);
            col.cx = 330; col.pszText = (LPWSTR)L"备注"; ListView_InsertColumn(g_gui.hFormatList, 4, &col);
            y += 185;

            // 进度条
            g_gui.hProgressBar = CreateWindowW(PROGRESS_CLASSW, L"",
                WS_CHILD | WS_VISIBLE | PBS_SMOOTH,
                10, y, 550, 22, hwnd, (HMENU)IDC_PROGRESS_BAR, nullptr, nullptr);
            SendMessageW(g_gui.hProgressBar, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
            y += 28;

            // 状态栏
            g_gui.hStatusText = CreateWindowW(L"STATIC", L"就绪 - 粘贴 URL 并点击 [获取信息]",
                WS_CHILD | WS_VISIBLE | SS_LEFT,
                10, y, 660, 20, hwnd, (HMENU)IDC_STATUS_TEXT, nullptr, nullptr);

            // 设置字体
            HFONT hFont = CreateFontW(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                CLEARTYPE_QUALITY, FF_DONTCARE, L"Microsoft YaHei UI");
            EnumChildWindows(hwnd, [](HWND child, LPARAM lf) -> BOOL {
                SendMessageW(child, WM_SETFONT, (WPARAM)(HFONT)lf, TRUE);
                return TRUE;
            }, (LPARAM)hFont);
            break;
        }

        case WM_COMMAND: {
            WORD id = LOWORD(wp);
            if (id == IDC_EXTRACT_BTN) gui_on_extract();
            else if (id == IDC_DOWNLOAD_BTN) gui_on_download();
            break;
        }

        case WM_TIMER: {
            // 定期更新进度
            if (wp == 1 && g_gui.progress) {
                gui_update_progress();
                if (g_gui.progress->finished) {
                    KillTimer(hwnd, 1);
                }
            }
            break;
        }

        case WM_DESTROY:
            if (g_gui.download_thread && g_gui.download_thread->joinable())
                g_gui.download_thread->detach();
            PostQuitMessage(0);
            break;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static int run_gui(HINSTANCE hInstance) {
    WNDCLASSEXW wc = {sizeof(WNDCLASSEXW)};
    wc.lpfnWndProc = gui_wndproc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"VideoDownloaderWindow";
    RegisterClassExW(&wc);

    int screen_w = GetSystemMetrics(SM_CXSCREEN);
    int screen_h = GetSystemMetrics(SM_CYSCREEN);
    int w = 700, h = 430;
    int x = (screen_w - w) / 2;
    int y = (screen_h - h) / 2;

    HWND hwnd = CreateWindowExW(0, L"VideoDownloaderWindow",
        L"万能视频下载器 v2.0 (C++ Native)",
        WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX),
        x, y, w, h, nullptr, nullptr, hInstance, nullptr);

    ShowWindow(hwnd, SW_SHOW);
    SetTimer(hwnd, 1, 200, nullptr);

    MSG msg = {};
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return (int)msg.wParam;
}

// ============================================================================
// CLI 模式
// ============================================================================

static void print_banner() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════╗
║        万能视频下载器 v2.0  (C++ Native Edition)          ║
║       对标 PyInstaller + yt-dlp + tkinter 版本            ║
╚══════════════════════════════════════════════════════════╝
)" << std::endl;
}

static void print_usage(const char* prog) {
    std::cout << "用法:\n"
              << "  " << prog << " [选项] <URL>\n"
              << "  " << prog << " --gui         启动图形界面\n\n"
              << "选项:\n"
              << "  -o, --output <path>     输出路径/模板 (默认: ./%(title)s.%(ext)s)\n"
              << "  -f, --format <id>       选择格式 (默认: best)\n"
              << "  -F, --list-formats      列出所有可用格式\n"
              << "  -q, --quality <height>  最高画质 (e.g. 720, 1080, 2160)\n"
              << "  -a, --audio-only        仅下载音频\n"
              << "  -p, --proxy <url>       代理 (e.g. http://127.0.0.1:7890)\n"
              << "  -c, --cookies <file>    Cookies 文件\n"
              << "  -t, --threads <n>       并发线程数 (默认: 4)\n"
              << "  -P, --playlist          下载播放列表\n"
              << "  -h, --help              显示帮助\n"
              << std::endl;
}

// ============================================================================
// main 入口
// ============================================================================

int main(int argc, char* argv[]) {
    // 初始化 COM 和通用控件
    INITCOMMONCONTROLSEX icex = {sizeof(INITCOMMONCONTROLSEX), ICC_PROGRESS_CLASS | ICC_LISTVIEW_CLASSES};
    InitCommonControlsEx(&icex);

    // 无参数启动 → GUI 模式
    if (argc == 1) {
        return run_gui(GetModuleHandleW(nullptr));
    }

    // 解析参数
    std::string url;
    std::string output = ".";
    std::string format = "best";
    std::string proxy;
    std::string cookies;
    int max_height = 1080;
    int threads = 4;
    bool list_formats = false;
    bool audio_only = false;
    bool playlist = false;
    bool gui_mode = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--gui") { gui_mode = true; }
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else if (arg == "-F" || arg == "--list-formats") { list_formats = true; }
        else if (arg == "-a" || arg == "--audio-only") { audio_only = true; }
        else if (arg == "-P" || arg == "--playlist") { playlist = true; }
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) { output = argv[++i]; }
        else if ((arg == "-f" || arg == "--format") && i + 1 < argc) { format = argv[++i]; }
        else if ((arg == "-q" || arg == "--quality") && i + 1 < argc) { max_height = atoi(argv[++i]); }
        else if ((arg == "-p" || arg == "--proxy") && i + 1 < argc) { proxy = argv[++i]; }
        else if ((arg == "-c" || arg == "--cookies") && i + 1 < argc) { cookies = argv[++i]; }
        else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) { threads = atoi(argv[++i]); }
        else if (arg[0] != '-') { url = arg; }
    }

    if (gui_mode) {
        return run_gui(GetModuleHandleW(nullptr));
    }

    if (url.empty()) {
        print_banner();
        print_usage(argv[0]);
        return 1;
    }

    print_banner();

    // 列表模式
    if (list_formats) {
        auto vi = YtDlpExtractor::extract(url, proxy, cookies);
        if (!vi) {
            std::cerr << "无法提取视频信息" << std::endl;
            return 1;
        }
        std::cout << "标题: " << vi->title << "\n";
        VideoDownloadManager::show_formats(*vi);
        return 0;
    }

    // 播放列表模式
    if (playlist) {
        auto vids = YtDlpExtractor::extract_playlist(url, proxy);
        if (!vids) {
            std::cerr << "无法提取播放列表" << std::endl;
            return 1;
        }
        std::cout << "播放列表: " << vids->size() << " 个视频\n\n";
        for (size_t i = 0; i < vids->size(); i++) {
            std::cout << "[" << (i + 1) << "/" << vids->size() << "] "
                      << (*vids)[i].title << std::endl;

            VideoDownloadManager::DownloadTask task;
            task.url = (*vids)[i].webpage_url.empty() ? url : (*vids)[i].webpage_url;
            task.output_dir = output;
            task.selected_format = format;
            task.max_height = max_height;
            task.num_threads = threads;
            task.proxy = proxy;
            task.cookies = cookies;
            task.audio_only = audio_only;

            if (!VideoDownloadManager::download(task)) {
                std::cerr << "  下载失败，继续下一个..." << std::endl;
            }
        }
        std::cout << "\n播放列表下载完成" << std::endl;
        return 0;
    }

    // 单视频下载
    VideoDownloadManager::DownloadTask task;
    task.url = url;
    task.output_dir = output;
    task.selected_format = format;
    task.max_height = max_height;
    task.num_threads = threads;
    task.proxy = proxy;
    task.cookies = cookies;
    task.audio_only = audio_only;

    bool ok = VideoDownloadManager::download(task);
    std::cout << (ok ? "\n下载完成!" : "\n下载失败") << std::endl;
    return ok ? 0 : 1;
}
