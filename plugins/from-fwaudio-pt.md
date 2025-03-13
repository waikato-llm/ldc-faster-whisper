# from-fwaudio-pt

* domain(s): pretrain
* generates: ldc.api.pretrain.PretrainData

Transcribes text from audio files (.wav, .mp3) to use for pretraining.

```
usage: from-fwaudio-pt [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                       [-N LOGGER_NAME] [-i [INPUT ...]] [-I [INPUT_LIST ...]]
                       [-1] [-m MODEL_SIZE] [-d DEVICE] [-c COMPUTE_TYPE]
                       [-b BEAM_SIZE]

Transcribes text from audio files (.wav, .mp3) to use for pretraining.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  -i [INPUT ...], --input [INPUT ...]
                        Path to the audio file(s) to read (.wav, .mp3); glob
                        syntax is supported; Supported placeholders: {HOME},
                        {CWD}, {TMP} (default: None)
  -I [INPUT_LIST ...], --input_list [INPUT_LIST ...]
                        Path to the text file(s) listing the audio files to
                        use (.wav, .mp3); Supported placeholders: {HOME},
                        {CWD}, {TMP} (default: None)
  -1, --combine_segments
                        Whether to combine the segments into a single document
                        or forward them one-by-one (default: False)
  -m MODEL_SIZE, --model_size MODEL_SIZE
                        The size of the whisper model to use, e.g., 'base' or
                        'large-v3' (default: base)
  -d DEVICE, --device DEVICE
                        The device to run on, e.g., 'cuda' or 'cpu' (default:
                        cpu)
  -c COMPUTE_TYPE, --compute_type COMPUTE_TYPE
                        The compute type to use, e.g., 'float16' or 'int8'
                        (default: int8)
  -b BEAM_SIZE, --beam_size BEAM_SIZE
                        The beam size to use for decoding (default: 5)
```
