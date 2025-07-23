import argparse
import sys
import os
import numpy as np
from pydub import AudioSegment

def main(args):
    """
    Applies the standardization step from the Emilia pipeline to a single audio file
    and saves the result.
    """
    # --- Add Emilia to Python Path ---
    # This allows us to import modules from the Emilia directory.
    emilia_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Emilia'))
    if emilia_path not in sys.path:
        sys.path.insert(0, emilia_path)

    # We can now import the necessary components from your pipeline
    from main import standardization
    from utils.tool import load_cfg

    print(f"Loading configuration from: {args.config_path}")
    cfg = load_cfg(args.config_path)

    print(f"Processing input file: {args.input_file}")

    # --- Run Standardization ---
    # This function performs normalization and resampling as defined in your pipeline.
    processed_audio = standardization(args.input_file, cfg)
    
    # The output is a dictionary containing the waveform and sample rate.
    waveform = processed_audio["waveform"]
    sample_rate = processed_audio["sample_rate"]

    print(f"  - Standardization complete.")
    print(f"  - New sample rate: {sample_rate} Hz")

    # --- Convert for Saving ---
    # The processed waveform is a float32 numpy array. We need to convert it
    # to 16-bit integers to be compatible with pydub for saving.
    waveform_int16 = (waveform * 32767).astype(np.int16)
    
    # Create a pydub AudioSegment from the raw waveform data.
    audio_segment = AudioSegment(
        waveform_int16.tobytes(), 
        frame_rate=sample_rate,
        sample_width=waveform_int16.dtype.itemsize,
        channels=1  # The standardization process creates mono audio
    )

    print(f"Saving standardized audio to: {args.output_file}")
    
    # Export the standardized audio to the specified output file.
    # The format is determined by the file extension (e.g., .wav, .mp3).
    output_format = os.path.splitext(args.output_file)[1].lstrip('.')
    audio_segment.export(args.output_file, format=output_format)

    print("\n✅ Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the standardization step of the Emilia pipeline."
    )
    parser.add_argument(
        "-i", "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input audio file."
    )
    parser.add_argument(
        "-o", "--output_file", 
        type=str, 
        required=True, 
        help="Path to save the standardized output audio file (e.g., output.wav)."
    )
    parser.add_argument(
        "-c", "--config_path", 
        type=str, 
        default="Emilia/config.json", 
        help="Path to the Emilia configuration file."
    )
    
    args = parser.parse_args()
    main(args) 