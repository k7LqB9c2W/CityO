#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "image_loader.h"

#include <windows.h>
#include <wincodec.h>

#include <string>

namespace {

std::wstring Utf8ToWide(const char* str) {
    if (!str || !str[0]) return {};
    int len = MultiByteToWideChar(CP_UTF8, 0, str, -1, nullptr, 0);
    if (len <= 1) return {};
    std::wstring out(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, str, -1, out.data(), len);
    out.pop_back();
    return out;
}

} // namespace

bool LoadImageRGBA(const char* path, std::vector<uint8_t>& outPixels, int& outW, int& outH) {
    outPixels.clear();
    outW = 0;
    outH = 0;
    if (!path || !path[0]) return false;

    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool coInit = (hr == S_OK || hr == S_FALSE);
    if (hr == RPC_E_CHANGED_MODE) {
        coInit = false;
    } else if (FAILED(hr)) {
        return false;
    }

    IWICImagingFactory* factory = nullptr;
    hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                          IID_IWICImagingFactory, (void**)&factory);
    if (FAILED(hr)) {
        if (coInit) CoUninitialize();
        return false;
    }

    std::wstring wpath = Utf8ToWide(path);
    if (wpath.empty()) {
        factory->Release();
        if (coInit) CoUninitialize();
        return false;
    }

    IWICBitmapDecoder* decoder = nullptr;
    hr = factory->CreateDecoderFromFilename(wpath.c_str(), nullptr, GENERIC_READ,
                                            WICDecodeMetadataCacheOnLoad, &decoder);
    if (FAILED(hr)) {
        factory->Release();
        if (coInit) CoUninitialize();
        return false;
    }

    IWICBitmapFrameDecode* frame = nullptr;
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr)) {
        decoder->Release();
        factory->Release();
        if (coInit) CoUninitialize();
        return false;
    }

    IWICFormatConverter* converter = nullptr;
    hr = factory->CreateFormatConverter(&converter);
    if (FAILED(hr)) {
        frame->Release();
        decoder->Release();
        factory->Release();
        if (coInit) CoUninitialize();
        return false;
    }

    hr = converter->Initialize(frame, GUID_WICPixelFormat32bppRGBA,
                               WICBitmapDitherTypeNone, nullptr, 0.0,
                               WICBitmapPaletteTypeCustom);
    if (FAILED(hr)) {
        converter->Release();
        frame->Release();
        decoder->Release();
        factory->Release();
        if (coInit) CoUninitialize();
        return false;
    }

    UINT w = 0;
    UINT h = 0;
    converter->GetSize(&w, &h);
    if (w == 0 || h == 0) {
        converter->Release();
        frame->Release();
        decoder->Release();
        factory->Release();
        if (coInit) CoUninitialize();
        return false;
    }

    outPixels.resize((size_t)w * (size_t)h * 4);
    hr = converter->CopyPixels(nullptr, w * 4, (UINT)outPixels.size(), outPixels.data());

    converter->Release();
    frame->Release();
    decoder->Release();
    factory->Release();
    if (coInit) CoUninitialize();

    if (FAILED(hr)) {
        outPixels.clear();
        return false;
    }

    outW = (int)w;
    outH = (int)h;
    return true;
}
