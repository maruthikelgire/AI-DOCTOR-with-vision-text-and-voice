import os
import gradio as gr
from pydub import AudioSegment
from brain_of_the_doctor import  encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

# System prompt for AI doctor
system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purposes.  
            What do you see in this image? Do you find anything wrong medically?  
            If there is a skin-related issue, mention the possible condition and suggest which type of specialist the patient should visit (e.g., dermatologist, plastic surgeon, etc.).  
            Do not add any numbers or special characters in your response.  
            Your response should be in one long paragraph. Always answer as if you are speaking to a real person.  
            Do not say 'In the image I see' but say 'With what I see, I think you have ....'  
            Provide a detailed explanation based on the query and medical insights, ensuring the response is as informative as needed without any strict length limitation."""


def process_inputs(audio_filepath, text_input, image_filepath):
    """
    Processes voice, text, and image input, then generates a doctor's response with voice output.
    
    Args:
        audio_filepath (str): Path to recorded audio file (if provided).
        text_input (str): User's typed input (if provided).
        image_filepath (str): Path to uploaded image file (if provided).
    
    Returns:
        tuple: (Speech-to-text output, Doctor's response, Generated voice response file)
    """
    # Convert speech to text if audio is provided
    if audio_filepath:
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )
    else:
        speech_to_text_output = ""  # No speech input

    # Use text input if provided
    final_user_input = text_input.strip() if text_input else speech_to_text_output

    if not final_user_input:
        return "No input provided", "Please provide text or voice input.", None

    # Process image if provided, otherwise process text/voice input
    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + " " + final_user_input,
            encoded_image=encode_image(image_filepath),
            model="llama-3.2-11b-vision-preview"
        )
    else:
        doctor_response = ("Based on your symptoms, I would suggest: "
                           "It's important to monitor your condition and consult a medical professional for accurate diagnosis. "
                           "Let me know more details about your symptoms for better guidance.")

    # Generate voice response
    voice_output_file = "final.mp3"
    text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath=voice_output_file)

    # Convert MP3 to WAV (better for Gradio compatibility)
    sound = AudioSegment.from_mp3(voice_output_file)
    sound.export("final.wav", format="wav")

    return speech_to_text_output, doctor_response, "final.wav"  # Use WAV for better compatibility

# Create the Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Voice Input (Optional)"),
        gr.Textbox(label="Text Input (Optional)"),
        gr.Image(type="filepath", label="Upload Medical Image (Optional)")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text Output"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice Response")
    ],
    title="AI Doctor Chatbot with Vision, Text, and Voice"
)

iface.launch(debug=True)
