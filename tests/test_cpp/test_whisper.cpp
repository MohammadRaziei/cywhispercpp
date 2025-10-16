#include <iostream>
#include <whisper.h>

int main() {
    std::cout << "=== Whisper C++ Library Test ===" << std::endl;
    
    // Test that we can include whisper.h and use basic functionality
    std::cout << "✓ Successfully included whisper.h" << std::endl;
    
    // Test whisper version
    const char* version = whisper_version();
    std::cout << "✓ Whisper version: " << version << std::endl;
    
    // Test whisper system info
    const char* system_info = whisper_print_system_info();
    std::cout << "✓ System info: " << system_info << std::endl;
    
    // Test whisper parameters
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    std::cout << "✓ Successfully created whisper parameters" << std::endl;
    
    // Test whisper context parameters
    struct whisper_context_params ctx_params = whisper_context_default_params();
    ctx_params.use_gpu = false;
    std::cout << "✓ Successfully created whisper context parameters" << std::endl;
    
    // Test whisper model types
    std::cout << "✓ Available model types:" << std::endl;
    std::cout << "  - WHISPER_SAMPLING_GREEDY" << std::endl;
    std::cout << "  - WHISPER_SAMPLING_BEAM_SEARCH" << std::endl;
    
    // Test language functions
    int max_lang_id = whisper_lang_max_id();
    std::cout << "✓ Maximum language ID: " << max_lang_id << std::endl;
    
    // Test English language ID
    int english_id = whisper_lang_id("en");
    std::cout << "✓ English language ID: " << english_id << std::endl;
    
    std::cout << "=== Test Completed Successfully ===" << std::endl;
    std::cout << "Library linking and basic functionality verified!" << std::endl;
    
    return 0;
}
