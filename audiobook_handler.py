import os
import json
import logging
import subprocess
import shutil
from typing import TypedDict

class PiperConfig(TypedDict):
    voices: dict[str, str]
    piper_binary: str
    ffmpeg_binary: str

class AudiobookGenerator:
    def __init__(self, config_path: str):
        self.log = logging.getLogger("AudiobookGenerator")
        self.config_path = config_path
        self.config_file = os.path.join(config_path, "piper_config.json")
        self.config: PiperConfig = self.load_config()

    def load_config(self) -> PiperConfig:
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.log.error(f"Failed to load config: {e}, using defaults")
        
        # Default configuration
        default_config: PiperConfig = {
            "voices": {
                "eng": os.path.expanduser("~/.local/share/piper_models/en_GB-alan-medium.onnx"),
                "ger": os.path.expanduser("~/.local/share/piper_models/de_DE-thorsten-medium.onnx"),
                "fre": os.path.expanduser("~/.local/share/piper_models/fr_FR-upmc-medium.onnx")
            },
            "piper_binary": "piper",
            "ffmpeg_binary": "ffmpeg"
        }
        self.save_config(default_config)
        return default_config

    def save_config(self, config: PiperConfig):
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            self.log.error(f"Failed to save config: {e}")

    def get_voice_for_language(self, language: str) -> str | None:
        # Simple mapping, can be improved to handle dialects/regions
        lang_base = language.split('_')[0].split('-')[0].lower()
        return self.config["voices"].get(lang_base)

    def generate_audiobook(self, text: str, language: str, output_path: str) -> bool:
        voice_model = self.get_voice_for_language(language)
        if not voice_model:
            self.log.error(f"No voice model configured for language: {language}")
            return False
        
        if not os.path.exists(voice_model):
             self.log.error(f"Voice model file not found: {voice_model}")
             return False

        piper_bin = self.config["piper_binary"]
        ffmpeg_bin = self.config["ffmpeg_binary"]

        if not shutil.which(piper_bin):
            self.log.error(f"Piper binary '{piper_bin}' not found in PATH")
            return False
        
        if not shutil.which(ffmpeg_bin):
            self.log.error(f"FFmpeg binary '{ffmpeg_bin}' not found in PATH")
            return False

        self.log.info(f"Generating audiobook in {language} using {voice_model}...")
        
        try:
            # Piper command: reads from stdin, outputs raw audio to stdout
            piper_cmd = [piper_bin, "--model", voice_model, "--output_raw"]
            
            # FFmpeg command: reads raw audio from stdin, encodes to mp3
            # Piper output is usually 22050Hz mono 16-bit PCM (s16le)
            ffmpeg_cmd = [
                ffmpeg_bin,
                "-f", "s16le",
                "-ar", "22050",
                "-ac", "1",
                "-i", "-",
                "-y", # Overwrite output
                output_path
            ]

            # Start processes
            piper_proc = subprocess.Popen(
                piper_cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.DEVNULL
            )
            
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd, 
                stdin=piper_proc.stdout, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )

            # Write text to piper's stdin
            if piper_proc.stdin:
                piper_proc.stdin.write(text.encode('utf-8'))
                piper_proc.stdin.close()

            # Wait for completion
            if piper_proc.stdout:
                piper_proc.stdout.close()
            
            ffmpeg_proc.wait()
            piper_proc.wait()

            if piper_proc.returncode == 0 and ffmpeg_proc.returncode == 0:
                self.log.info(f"Audiobook saved to {output_path}")
                return True
            else:
                self.log.error(f"Generation failed. Piper: {piper_proc.returncode}, FFmpeg: {ffmpeg_proc.returncode}")
                return False

        except Exception as e:
            self.log.error(f"Error generating audiobook: {e}")
            return False
