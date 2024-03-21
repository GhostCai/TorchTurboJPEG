#include <iostream>
#include <nvjpeg.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using tensor = torch::Tensor;

class Encoder {
private:
    nvjpegHandle_t h;
    nvjpegJpegState_t s;
    nvjpegEncoderState_t es;
    nvjpegEncoderParams_t p;
    tensor buf;
    nvjpegImage_t img;

public:
    Encoder(int q, int bs) {
        // q: quality, bs: buffer size
        std::cout << "Warning: This is a dangerous encoder which will NOT check for buffer overflows, use at your own risk\n";
        std::cout << "TIPS: Make sure input tensor is (a)CUDA (b)contiguous (c)uint8 (d)3D (e)3x512x512\n";
        cudaStream_t stream = nullptr;
        buf = torch::empty({bs}, torch::TensorOptions().dtype(torch::kUInt8));
        nvjpegCreateSimple(&h);
        nvjpegJpegStateCreate(h, &s);
        nvjpegEncoderStateCreate(h, &es, NULL);
        nvjpegEncoderParamsCreate(h, &p, stream);
        nvjpegEncoderParamsSetQuality(p, q, stream);
        nvjpegEncoderParamsSetOptimizedHuffman(p, 1, stream);
        nvjpegEncoderParamsSetSamplingFactors(p, nvjpegChromaSubsampling_t::NVJPEG_CSS_420, stream);
    }

    ~Encoder() {
        nvjpegEncoderParamsDestroy(p);
        nvjpegEncoderStateDestroy(es);
        nvjpegJpegStateDestroy(s);
        nvjpegDestroy(h);
    }

    tensor encode(tensor input) {
        py::gil_scoped_release release;
        size_t len,plane_stride = at::stride(input, 0);
        unsigned int plane_stride1 = (unsigned int)at::stride(input, 1);
        unsigned char* data_ptr = (unsigned char*)input.data_ptr();
        img.pitch[0] = plane_stride1;
        img.channel[0] = data_ptr + plane_stride * 0;
        img.pitch[1] = plane_stride1;
        img.channel[1] = data_ptr + plane_stride * 1;
        img.pitch[2] = plane_stride1;
        img.channel[2] = data_ptr + plane_stride * 2;
        nvjpegEncodeImage(h, es, p, &img, nvjpegInputFormat_t::NVJPEG_INPUT_RGB, 512, 512, nullptr);
        nvjpegEncodeRetrieveBitstream(h, es, (unsigned char*)buf.data_ptr(), &len, nullptr);
        return buf.slice(0, 0, len);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Encoder>(m, "Encoder")
        .def(py::init<int, int>())
        .def("encode", &Encoder::encode);
}
