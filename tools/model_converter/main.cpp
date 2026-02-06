#include <iostream>
#include <string>

void print_help() {
    std::cout << "VisionEngine Model Converter" << std::endl;
    std::cout << "Usage: model_converter <command> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  convert   - Convert model format" << std::endl;
    std::cout << "  quantize  - Quantize model to INT8" << std::endl;
    std::cout << "  info      - Show model information" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --input   - Input model path" << std::endl;
    std::cout << "  --output  - Output model path" << std::endl;
    std::cout << "  --format  - Output format (onnx, tensorrt)" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help();
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "--help" || command == "-h") {
        print_help();
        return 0;
    }
    
    std::cout << "VisionEngine Tools v1.0.0" << std::endl;
    return 0;
}
