# distutils: language = c++
# cython: language_level = 3

"""Cython bindings for whisper.cpp"""

from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, unique_ptr
from cpython.ref cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from "whisper.h":
    ctypedef struct whisper_context:
        pass
    
    ctypedef struct whisper_state:
        pass
    
    ctypedef struct whisper_full_params:
        int n_threads
        int n_max_text_ctx
        int offset_ms
        int duration_ms
        bool translate
        bool no_context
        bool single_segment
        bool print_special
        bool print_progress
        bool print_realtime
        bool print_timestamps
        bool token_timestamps
        float thold_pt
        float thold_ptsum
        int max_len
        bool split_on_word
        int max_tokens
        bool speed_up
        int audio_ctx
        int vad_thold
        int freq_thold
        bool suppress_blank
        bool suppress_non_speech_tokens
        float temperature
        float max_initial_ts
        float length_penalty
        float temperature_inc
        float entropy_thold
        float logprob_thold
        float no_speech_thold
        bool greedy_best_of
        bool beam_search_best_of
        int beam_size
        float patience
        int new_segment_callback
        int new_segment_callback_user_data
        int progress_callback
        int progress_callback_user_data
        int encoder_begin_callback
        int encoder_begin_callback_user_data
        int logits_filter_callback
        int logits_filter_callback_user_data
    
    ctypedef struct whisper_token_data:
        int id
        float p
        float plog
        float pt
        float ptsum
        int64_t t0
        int64_t t1
        float vlen
    
    ctypedef struct whisper_segment:
        int64_t t0
        int64_t t1
        const char* text
        vector[whisper_token_data] tokens
    
    whisper_context* whisper_init_from_file(const char* path_model)
    whisper_context* whisper_init_from_buffer(void* buffer, size_t buffer_size)
    void whisper_free(whisper_context* ctx)
    
    int whisper_pcm_to_mel(
        whisper_context* ctx,
        const float* samples,
        int n_samples,
        int n_threads
    )
    
    int whisper_pcm_to_mel_phase_vocoder(
        whisper_context* ctx,
        const float* samples,
        int n_samples,
        int n_threads
    )
    
    int whisper_set_mel(
        whisper_context* ctx,
        const float* data,
        int n_len,
        int n_mel
    )
    
    int whisper_encode(
        whisper_context* ctx,
        int offset,
        int n_threads
    )
    
    int whisper_decode(
        whisper_context* ctx,
        const whisper_token* tokens,
        int n_tokens,
        int n_past,
        int n_threads
    )
    
    int whisper_tokenize(
        whisper_context* ctx,
        const char* text,
        whisper_token* tokens,
        int n_max_tokens
    )
    
    int whisper_lang_id(const char* lang)
    
    int whisper_n_len(whisper_context* ctx)
    int whisper_n_vocab(whisper_context* ctx)
    int whisper_n_text_ctx(whisper_context* ctx)
    int whisper_n_audio_ctx(whisper_context* ctx)
    int whisper_is_multilingual(whisper_context* ctx)
    
    float* whisper_get_logits(whisper_context* ctx)
    
    const char* whisper_token_to_str(whisper_context* ctx, whisper_token token)
    whisper_token whisper_token_eot(whisper_context* ctx)
    whisper_token whisper_token_sot(whisper_context* ctx)
    whisper_token whisper_token_solm(whisper_context* ctx)
    whisper_token whisper_token_prev(whisper_context* ctx)
    whisper_token whisper_token_nosp(whisper_context* ctx)
    whisper_token whisper_token_not(whisper_context* ctx)
    whisper_token whisper_token_beg(whisper_context* ctx)
    whisper_token whisper_token_lang(whisper_context* ctx, int lang_id)
    
    int whisper_full(
        whisper_context* ctx,
        whisper_full_params params,
        const float* samples,
        int n_samples
    )
    
    int whisper_full_parallel(
        whisper_context* ctx,
        whisper_full_params params,
        const float* samples,
        int n_samples,
        int n_processors
    )
    
    int whisper_full_n_segments(whisper_context* ctx)
    int64_t whisper_full_get_segment_t0(whisper_context* ctx, int i_segment)
    int64_t whisper_full_get_segment_t1(whisper_context* ctx, int i_segment)
    const char* whisper_full_get_segment_text(whisper_context* ctx, int i_segment)
    
    int whisper_full_n_tokens(whisper_context* ctx, int i_segment)
    whisper_token_data whisper_full_get_token_data(whisper_context* ctx, int i_segment, int i_token)
    const char* whisper_full_get_token_text(whisper_context* ctx, int i_segment, int i_token)
    
    whisper_full_params whisper_full_default_params(whisper_context* ctx, int strategy)
    
    const char* whisper_print_system_info()

cdef class WhisperParams:
    cdef whisper_full_params params
    
    def __cinit__(self):
        pass
    
    def __init__(self):
        pass

cdef class WhisperState:
    cdef whisper_state* state
    
    def __cinit__(self):
        pass
    
    def __init__(self):
        pass

cdef class Whisper:
    cdef whisper_context* ctx
    
    def __cinit__(self):
        self.ctx = NULL
    
    def __dealloc__(self):
        if self.ctx != NULL:
            whisper_free(self.ctx)
    
    def __init__(self, model_path: str):
        """Initialize whisper from model file.
        
        Args:
            model_path: Path to the whisper model file
        """
        cdef bytes model_path_bytes = model_path.encode('utf-8')
        cdef const char* model_path_cstr = model_path_bytes
        self.ctx = whisper_init_from_file(model_path_cstr)
        if self.ctx == NULL:
            raise RuntimeError(f"Failed to load whisper model from {model_path}")
    
    @staticmethod
    def init_from_buffer(buffer: bytes):
        """Initialize whisper from buffer.
        
        Args:
            buffer: Model data as bytes
        
        Returns:
            Whisper instance
        """
        cdef Whisper instance = Whisper.__new__(Whisper)
        cdef char* buffer_ptr = <char*>buffer
        instance.ctx = whisper_init_from_buffer(buffer_ptr, len(buffer))
        if instance.ctx == NULL:
            raise RuntimeError("Failed to load whisper model from buffer")
        return instance
    
    def transcribe(self, audio_data, params: WhisperParams = None):
        """Transcribe audio data.
        
        Args:
            audio_data: Audio samples as list or numpy array of floats
            params: Whisper parameters (optional)
        
        Returns:
            List of transcribed segments
        """
        # This is a simplified implementation
        # In a real implementation, we would handle the audio data conversion
        # and call whisper_full with appropriate parameters
        pass
    
    def get_system_info(self) -> str:
        """Get system information.
        
        Returns:
            System information string
        """
        cdef const char* info = whisper_print_system_info()
        if info == NULL:
            return ""
        return info.decode('utf-8')
    
    def is_multilingual(self) -> bool:
        """Check if model is multilingual.
        
        Returns:
            True if multilingual, False otherwise
        """
        return whisper_is_multilingual(self.ctx) != 0
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Vocabulary size
        """
        return whisper_n_vocab(self.ctx)
    
    def get_audio_ctx_size(self) -> int:
        """Get audio context size.
        
        Returns:
            Audio context size
        """
        return whisper_n_audio_ctx(self.ctx)
    
    def get_text_ctx_size(self) -> int:
        """Get text context size.
        
        Returns:
            Text context size
        """
        return whisper_n_text_ctx(self.ctx)
