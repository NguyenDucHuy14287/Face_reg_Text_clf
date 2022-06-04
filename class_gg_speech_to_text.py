from google.cloud import speech_v1p1beta1
import os
import io

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'bk-tiktok-29e9951ff1b8.json'

def speech_to_text(file):
    # Creates google client
    client = speech_v1p1beta1.SpeechClient()

    # Full path of the audio file, Replace with your file name
    file_name = os.path.join(os.path.dirname(__file__), "mp3/" + file)

    # Loads the audio file into memory
    with io.open(file_name, "rb") as audio_file:
        content = audio_file.read()
        audio = speech_v1p1beta1.RecognitionAudio(content=content)

    config = speech_v1p1beta1.RecognitionConfig(
        encoding=speech_v1p1beta1.RecognitionConfig.AudioEncoding.MP3,
        audio_channel_count=2,
        language_code="vi-VN",
        alternative_language_codes=["en-US", "vi-VN"],
        model="command_and_search",
        sample_rate_hertz=16000,
    )

    try:
        # Sends the request to google to transcribe the audio
        response = client.recognize(request={"config": config, "audio": audio})

        # Reads the response
        _text = ""
        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].transcript))
            _text = _text + " " + format(result.alternatives[0].transcript)
        return _text
    except:
        return ''

text = speech_to_text('7086795403257072411.mp3')
print('text')
