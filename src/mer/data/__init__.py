"""Dataset loading, transcript parsing, and label normalization helpers."""

from mer.data.labels import (
    CANONICAL_EMOTION_SET,
    CANONICAL_EMOTIONS,
    DEFAULT_SYNONYMS,
    normalize_emotion_name,
    normalize_prediction_text,
)
from mer.data.esd import (
    ESD_ENGLISH_SPEAKER_MAX,
    ESD_ENGLISH_SPEAKER_MIN,
    esd_transcript_paths,
    is_english_esd_speaker_path,
    read_esd_transcript,
    resolve_esd_wav_path,
    speaker_id_from_esd_path,
    utterance_id_from_esd_path,
)
from mer.data.iemocap import (
    infer_gender_from_utt,
    infer_session_from_utt,
    resolve_iemocap_audio_path,
)

__all__ = [
    "CANONICAL_EMOTION_SET",
    "CANONICAL_EMOTIONS",
    "DEFAULT_SYNONYMS",
    "ESD_ENGLISH_SPEAKER_MAX",
    "ESD_ENGLISH_SPEAKER_MIN",
    "esd_transcript_paths",
    "infer_gender_from_utt",
    "infer_session_from_utt",
    "is_english_esd_speaker_path",
    "normalize_emotion_name",
    "normalize_prediction_text",
    "read_esd_transcript",
    "resolve_esd_wav_path",
    "resolve_iemocap_audio_path",
    "speaker_id_from_esd_path",
    "utterance_id_from_esd_path",
]
