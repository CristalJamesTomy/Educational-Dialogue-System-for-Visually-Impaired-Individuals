This project is a web-based interactive system that processes images of Finite State Automata (FSA) diagrams, extracts their structure (states, transitions, and labels) using computer vision techniques (OpenCV and Tesseract), and provides a voice-enabled chatbot interface (using speech recognition and text-to-speech) to answer questions about the FSA's components, behavior, and basic concepts, helping users understand and analyze the automaton through both visual and auditory feedback.

1. Core Functionality

Processes uploaded images of Finite State Automata (FSA) diagrams.

Uses computer vision (OpenCV) to detect states (circles), transitions (lines), and labels.

Identifies initial states (with arrows) and final states (double circles).

Extracts transition labels (e.g., 'a', 'b') using OCR (Tesseract) and pattern recognition.

2. Interactive Chatbot

Provides a voice-enabled chatbot using speech recognition (Google Speech API) and text-to-speech (pyttsx3).

Answers questions about:

States (initial, final, normal).

Transitions (source, destination, input symbols).

Basic FSA concepts (what is a state, transition, etc.).

3. Visualization & Debugging
   
Generates an annotated output image highlighting detected states and transitions.

Displays debugging info (bounding boxes, labels, and detected elements).

4. User Interaction
   
Web-based (Flask) for easy access.

Supports text input and voice commands.

Gives spoken responses for accessibility.

5. Technical Stack
   
Backend: Python (Flask, OpenCV, Pytesseract).

Speech Processing: speech_recognition, pyttsx3.

Frontend: HTML, JavaScript (for voice/text interaction).

6. Use Cases
   
Helps students learn FSA concepts interactively.

Assists in automated FSA diagram analysis (e.g., for academic projects).

Useful for accessibility (voice-based learning).
