import streamlit as st
import whisper
import tempfile
import os
import yt_dlp as youtube_dl
import ssl
import pandas as pd
from io import BytesIO
import numpy as np
from pyannote.audio import Pipeline

ssl._create_default_https_context = ssl._create_unverified_context

# Replace with your own secure username and password
USERNAME = "test"
PASSWORD = "test123"

# Replace with your Hugging Face API token
HUGGINGFACE_TOKEN = "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"

# Path to cookies.txt file on your local machine
COOKIES_PATH = '/Users/tcolo/downloads/transcription/cookies.txt'

class YouTubeTranscriber:
    def __init__(self, huggingface_token):
        self.huggingface_token = huggingface_token
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=huggingface_token)

    @staticmethod
    def download_youtube_audio(url, cookies_path=None):
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio_mp4 = temp_audio_path.name.replace('.wav', '.mp4')
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_audio_mp4,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'cookiefile': cookies_path,
            'ffmpeg_location': ffmpeg_path,
            'keepvideo': True,
            'noplaylist': True
        }

        try:
            if os.path.exists(temp_audio_mp4):
                os.remove(temp_audio_mp4)

            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            if os.path.getsize(temp_audio_mp4) == 0:
                raise Exception("Downloaded file is empty.")

            conversion_command = f'{ffmpeg_path} -y -i "{temp_audio_mp4}" "{temp_audio_path.name}"'
            conversion_result = os.system(conversion_command)
            if conversion_result != 0:
                raise Exception("ffmpeg conversion failed.")
        except Exception as e:
            st.error(f"Error during download or conversion: {str(e)}")
            raise
        finally:
            if os.path.exists(temp_audio_mp4):
                os.remove(temp_audio_mp4)

        return temp_audio_path.name

    def transcribe_audio(self, audio_path):
        # Load the Whisper model with GPU support
        model = whisper.load_model("large", device="cuda")
        transcription = model.transcribe(audio_path)
        return transcription

    def diarize_audio(self, audio_path, num_speakers):
        try:
            diarization = self.pipeline({"uri": "filename", "audio": audio_path}, num_speakers=num_speakers)
            return diarization
        except Exception as e:
            st.error(f"Error during diarization: {str(e)}")
            return []

    @staticmethod
    def convert_diarization_to_segments(diarization):
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        return segments

    @staticmethod
    def annotate_transcription_with_speakers(transcription, segments, speaker_labels):
        annotated_transcription = []
        for item in transcription['segments']:
            start_time = item['start']
            end_time = item['end']
            speaker = "Unknown"
            for segment in segments:
                segment_start, segment_end, segment_speaker = segment
                if start_time >= segment_start and end_time <= segment_end:
                    speaker = speaker_labels.get(segment_speaker, segment_speaker)
                    break
            if speaker == "Unknown":
                speaker = "SPEAKER_UNKNOWN"
            annotated_transcription.append(f"{speaker}: {item['text']}")
        return annotated_transcription

def check_password():
    def password_entered():
        if st.session_state["username"] == USERNAME and st.session_state["password"] == PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.sidebar.text_input("Username", key="username")
        st.sidebar.text_input("Password", type="password", key="password")
        st.sidebar.button("Login", on_click=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        st.sidebar.text_input("Username", key="username")
        st.sidebar.text_input("Password", type="password", key="password")
        st.sidebar.error("Username or password is incorrect")
        st.sidebar.button("Login", on_click=password_entered)
        return False
    else:
        return True

if check_password():
    ssl._create_default_https_context = ssl._create_unverified_context

    ffmpeg_path = '/Users/tcolo/scoop/apps/ffmpeg/current/bin/ffmpeg.exe'
    ffprobe_path = '/Users/tcolo/scoop/apps/ffmpeg/current/bin/ffprobe.exe'

    os.environ['PATH'] += os.pathsep + os.path.dirname(ffmpeg_path)
    os.environ['PATH'] += os.pathsep + os.path.dirname(ffprobe_path)

    assert os.path.isfile(ffmpeg_path), f"ffmpeg not found at {ffmpeg_path}"
    assert os.path.isfile(ffprobe_path), f"ffprobe not found at {ffprobe_path}"

    st.title("YouTube Video Transcription and Speaker Identification")
    st.write("Upload a YouTube video URL, video ID, or a CSV file with YouTube URLs or video IDs for transcription.")

    with st.sidebar:
        st.header("Select input method")
        option = st.radio("Input Method", ["YouTube URL", "Video ID", "CSV Upload"])

        if option == "YouTube URL":
            youtube_url = st.text_input("Enter YouTube Video URL")
        elif option == "Video ID":
            video_id = st.text_input("Enter YouTube Video ID")
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        elif option == "CSV Upload":
            csv_file = st.file_uploader("Upload CSV with YouTube URLs or Video IDs", type=["csv"])
            if csv_file is not None:
                df = pd.read_csv(csv_file)
                youtube_url = df.iloc[0, 0]

        num_speakers = st.number_input("Enter the number of speakers", min_value=1, step=1)

    progress_placeholder = st.empty()

    transcriber = YouTubeTranscriber(HUGGINGFACE_TOKEN)

    if "transcription" not in st.session_state:
        st.session_state.transcription = None

    if "annotated_transcription" not in st.session_state:
        st.session_state.annotated_transcription = None

    def transcribe_and_diarize():
        try:
            with st.spinner('Downloading audio...'):
                progress_placeholder.progress(0)
                audio_file_path = transcriber.download_youtube_audio(youtube_url, COOKIES_PATH)
                progress_placeholder.progress(33)

            with st.spinner('Transcribing audio...'):
                transcription = transcriber.transcribe_audio(audio_file_path)
                progress_placeholder.progress(66)

                transcription_text = "\n".join([f"{seg['text']}" for seg in transcription['segments']])
                st.session_state.transcription = transcription_text

                # Display raw transcription immediately
                st.text_area("Raw Transcription", st.session_state.transcription, height=300, key="raw_transcription_display")

            with st.spinner('Performing speaker diarization...'):
                diarization = transcriber.diarize_audio(audio_file_path, num_speakers)
                progress_placeholder.progress(100)

                segments = transcriber.convert_diarization_to_segments(diarization)
                speaker_labels = {f"SPEAKER_{i:02d}": f"SPEAKER_{i:02d}" for i in range(num_speakers)}
                annotated_transcription = transcriber.annotate_transcription_with_speakers(transcription, segments, speaker_labels)

                annotated_transcription_text = "\n".join(annotated_transcription)
                st.session_state.annotated_transcription = annotated_transcription_text

                # Display annotated transcription
                st.text_area("Annotated Transcription", st.session_state.annotated_transcription, height=300, key="annotated_transcription_display")

                os.remove(audio_file_path)
                st.success("Transcription and Speaker Identification complete")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    if st.sidebar.button("Transcribe YouTube Video"):
        if youtube_url:
            transcribe_and_diarize()
        else:
            st.error("Please provide a valid input.")

    if st.session_state.transcription:
        st.text_area("Raw Transcription", st.session_state.transcription, height=300, key="raw_transcription_display_final")

    if st.session_state.annotated_transcription:
        st.text_area("Annotated Transcription", st.session_state.annotated_transcription, height=300, key="annotated_transcription_display_final-1")

        # Input fields for custom speaker names
        st.write("## Assign Custom Names to Speakers")
        speaker_names = {}
        for i in range(num_speakers):
            speaker_label = f"SPEAKER_{i:02d}"
            speaker_names[speaker_label] = st.text_input(f"Custom name for {speaker_label}", value=speaker_label)

        # Apply custom names to annotated transcription
        if st.button("Apply Custom Names"):
            annotated_transcription = st.session_state.annotated_transcription.split('\n')
            updated_transcription = []
            for line in annotated_transcription:
                speaker, text = line.split(": ", 1)
                custom_name = speaker_names.get(speaker, speaker)
                updated_transcription.append(f"{custom_name}: {text}")
            st.session_state.annotated_transcription = "\n".join(updated_transcription)
            st.success("Custom names applied successfully")

        st.text_area("Edit Annotated Transcription", st.session_state.annotated_transcription, height=300, key="edit_annotated_transcription_display")

        if st.button("Save Changes"):
            st.session_state.annotated_transcription = st.session_state.edit_annotated_transcription_display
            st.success("Changes saved successfully")

        if st.session_state.annotated_transcription:
            st.download_button("Download Annotated Transcription", data=st.session_state.annotated_transcription, file_name="annotated_transcription.txt", mime="text/plain")
