from youtube_transcript_api import YouTubeTranscriptApi

def getTranscripts(id):
    srt = YouTubeTranscriptApi.get_transcript(id,
                                          languages=['en'])
    transcript = []
    for i in srt:
        transcript.append(i['text'])
    
    return " ".join(transcript)