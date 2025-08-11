from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pytesseract
import speech_recognition as sr
from typing import Tuple, List, Dict
import math
import pyttsx3
import os

app = Flask(__name__)



def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
   
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
   
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return image, thresh

def detect_circles(image: np.ndarray, thresh: np.ndarray) -> List[Dict]:
 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,  
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100
    )
    
    states = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            states.append({
                'center': center,
                'radius': radius,
                'bbox': (
                    int(center[0] - radius),
                    int(center[1] - radius),
                    int(radius * 2),
                    int(radius * 2)
                )
            })
    
    return states

def detect_initial_state_arrow(thresh: np.ndarray, states: List[Dict]) -> Dict:
   
    leftmost_state = None
    min_x = float('inf')
    
    for state in states:
        center = state['center']
        radius = state['radius']
        x, y = center
        
       
        if x < min_x:
            min_x = x
            leftmost_state = state
    
    if leftmost_state:
        x, y = leftmost_state['center']
        radius = leftmost_state['radius']
        
        
        search_region = max(0, x - int(radius * 3))
        roi_left = thresh[
            max(0, y - radius):min(thresh.shape[0], y + radius),
            search_region:max(0, x - radius)
        ]
        
        if roi_left.size > 0:
            edges = cv2.Canny(roi_left, 50, 150)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, threshold=20,
                minLineLength=radius, maxLineGap=10
            )
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1))
                    if angle < np.pi/6:  
                        return leftmost_state
    
    return None

def detect_double_circles(thresh: np.ndarray, states: List[Dict]) -> Dict:
   
    best_final_state = None
    max_circle_score = 0
    
    for state in states:
        center = state['center']
        radius = state['radius']
        
       
        inner_mask = np.zeros(thresh.shape, dtype=np.uint8)
        outer_mask = np.zeros(thresh.shape, dtype=np.uint8)
        
        cv2.circle(inner_mask, center, int(radius - 5), 255, 2)
        cv2.circle(outer_mask, center, int(radius + 5), 255, 2)
        
        
        inner_pixels = cv2.bitwise_and(thresh, inner_mask)
        outer_pixels = cv2.bitwise_and(thresh, outer_mask)
        
        inner_count = np.count_nonzero(inner_pixels)
        outer_count = np.count_nonzero(outer_pixels)
       
        circle_score = inner_count + outer_count
        
       
        if circle_score > max_circle_score:
            max_circle_score = circle_score
            best_final_state = state
    
    
    if best_final_state:
        center = best_final_state['center']
        radius = best_final_state['radius']
        
        
        outer_mask = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.circle(outer_mask, center, radius + 5, 255, 2)
        
       
        inner_mask = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.circle(inner_mask, center, radius - 5, 255, 2)
        
       
        outer_contours, _ = cv2.findContours(outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inner_contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        if len(outer_contours) == 1 and len(inner_contours) == 1:
            return best_final_state
    
    return None

def detect_transitions(thresh: np.ndarray, states: List[Dict]) -> List[Dict]:
   
    transitions = []
    processed_pairs = set() 
    
    
    state_mask = np.ones(thresh.shape, dtype=np.uint8) * 255
    for state in states:
        cv2.circle(state_mask, state['center'], state['radius'] + 5, 0, -1)
    
    
    masked_thresh = cv2.bitwise_and(thresh, state_mask)
    
   
    lines = cv2.HoughLinesP(
        masked_thresh,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=30,
        maxLineGap=20
    )
    
    if lines is None:
        return transitions
    
   
    sorted_states = sorted(states, key=lambda s: s['center'][0])
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
       
        if x2 < x1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        
      
        source_state = None
        dest_state = None
        min_source_dist = float('inf')
        min_dest_dist = float('inf')
        
        for state in sorted_states:
            center = state['center']
            radius = state['radius']
            
           
            dist_to_start = np.sqrt((center[0] - x1)**2 + (center[1] - y1)**2)
            dist_to_end = np.sqrt((center[0] - x2)**2 + (center[1] - y2)**2)
            
            
            if abs(dist_to_start - radius) < 20:
                if dist_to_start < min_source_dist:
                    min_source_dist = dist_to_start
                    source_state = state
            
            if abs(dist_to_end - radius) < 20:
                if dist_to_end < min_dest_dist:
                    min_dest_dist = dist_to_end
                    dest_state = state
        
      
        if source_state and dest_state and source_state != dest_state:
            state_pair = (states.index(source_state), states.index(dest_state))
            if state_pair not in processed_pairs:
                processed_pairs.add(state_pair)
                
               
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                
               
                midpoint_x = (x1 + x2) // 2
                midpoint_y = (y1 + y2) // 2
                
              
                perpendicular_x = midpoint_x
                perpendicular_y = midpoint_y - 20  
                
                transitions.append({
                    'source': source_state,
                    'destination': dest_state,
                    'line_points': (x1, y1, x2, y2),
                    'midpoint': (midpoint_x, midpoint_y),
                    'label_position': (perpendicular_x, perpendicular_y),
                    'angle': angle_rad,
                    'source_idx': sorted_states.index(source_state),
                    'dest_idx': sorted_states.index(dest_state)
                })
    
    return transitions

def detect_small_character(image: np.ndarray, center_x: int, center_y: int, search_radius: int = 30) -> str:
   
    x1 = max(0, center_x - search_radius)
    y1 = max(0, center_y - search_radius)
    x2 = min(image.shape[1], center_x + search_radius)
    y2 = min(image.shape[0], center_y)  
   
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return ""
    
  
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi.copy()
    
   
    _, binary1 = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY_INV)
    binary2 = cv2.adaptiveThreshold(
        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    
    for binary in [binary1, binary2]:
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
           
            if cv2.contourArea(cnt) > 5 and cv2.contourArea(cnt) < 500:
                x, y, w, h = cv2.boundingRect(cnt)
                
               
                aspect_ratio = float(w) / h
                
               
                char_img = binary[y:y+h, x:x+w]
                
            
                if h > 1: 
                    upper_half = char_img[:h//2, :]
                    lower_half = char_img[h//2:, :]
                    
                    upper_pixel_count = np.sum(upper_half > 0)
                    lower_pixel_count = np.sum(lower_half > 0)
                     
                    if w > 1:  
                        left_half = char_img[:, :w//2]
                        right_half = char_img[:, w//2:]
                        
                        left_pixel_count = np.sum(left_half > 0)
                        right_pixel_count = np.sum(right_half > 0)
                        
                      
                        if (0.8 <= aspect_ratio <= 1.2 and
                            abs(upper_pixel_count - lower_pixel_count) < 10 and
                            abs(left_pixel_count - right_pixel_count) < 10):
                            return "a"
                        
                    
                        elif (aspect_ratio < 0.6 and
                              upper_pixel_count > lower_pixel_count and
                              abs(left_pixel_count - right_pixel_count) < 10):
                            return "b"
                        
                      
                        elif x < roi.shape[1] // 2:  
                            return "0"
                        else:
                            return "1"
    
   
    return ""

def extract_transition_label(original_image: np.ndarray, transition: Dict, transition_idx: int, total_transitions: int) -> str:
    
   
    label_x, label_y = transition['label_position']
    
   
    label = detect_small_character(original_image, label_x, label_y)
    
    
    if not label:
        
        if total_transitions > 1:
          
            if transition_idx == 0:
                label = "a"
            elif transition_idx == 1:
                label = "b"
            else:
               
                label = chr(ord('a') + transition_idx)
    
    return label

def process_fsa_image(image_path: str) -> Dict:
   
    try:
      
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError("Could not load image")
        
        image, thresh = preprocess_image(image_path)
        
    
        states = detect_circles(image, thresh)
        if not states:
            raise ValueError("No states detected in the image")
        
        initial_state = detect_initial_state_arrow(thresh, states)
        fina
        reordered_states = []
        if initial_state:
            reordered_states.append(initial_state)
        for state in states:
            if state != initial_state and state != final_state:
                reordered_states.append(state)
        if final_state:
            reordered_states.append(final_state)
        transitions = detect_transitions(thresh, reordered_states)
        
      
        transitions.sort(key=lambda t: t['source']['center'][0])
        
       
        result_image = original_image.copy()
        
     
        state_info = []
        for i, state in enumerate(reordered_states):
            center = state['center']
            radius = state['radius']
            
            # Determine state type
            state_types = []
            if state == initial_state:
                state_types.append("Initial")
            if state == final_state:
                state_types.append("Final")
            if not state_types:
                state_types.append("Normal")
            
            # Draw state
            cv2.circle(result_image, center, radius, (0, 255, 0), 2)
            if state == final_state:
                cv2.circle(result_image, center, radius - 5, (0, 255, 0), 2)
            
           
            label_text = f"S{i} ({', '.join(state_types)})"
            cv2.putText(result_image, label_text,
                        (center[0] - 20, center[1] - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
           
            state_info.append({
                "id": i,
                "type": state_types,
                "center": (int(center[0]), int(center[1])), 
                "radius": int(radius) 
            })
        
     
        transition_data = []
        for i, transition in enumerate(transitions):
            x1, y1, x2, y2 = transition['line_points']
            
          
            cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
          
            angle = transition['angle']
            arrow_length = 15
            arrow_angle = np.pi / 6
            
            x_end, y_end = x2, y2
            x_arrow1 = int(x_end - arrow_length * np.cos(angle + arrow_angle))
            y_arrow1 = int(y_end - arrow_length * np.sin(angle + arrow_angle))
            x_arrow2 = int(x_end - arrow_length * np.cos(angle - arrow_angle))
            y_arrow2 = int(y_end - arrow_length * np.sin(angle - arrow_angle))
            
            cv2.line(result_image, (x_end, y_end), (x_arrow1, y_arrow1), (255, 0, 0), 2)
            cv2.line(result_image, (x_end, y_end), (x_arrow2, y_arrow2), (255, 0, 0), 2)
            
          
            label = extract_transition_label(original_image, transition, i, len(transitions))
            
  
            label_pos = transition['label_position']
            cv2.circle(result_image, label_pos, 5, (0, 0, 255), -1)
            
            
            search_radius = 30
            cv2.rectangle(result_image, 
                         (label_pos[0] - search_radius, label_pos[1] - search_radius),
                         (label_pos[0] + search_radius, label_pos[1]),
                         (0, 255, 255), 1)
            
         
            cv2.putText(result_image, f"T{i}: '{label}'",
                        (transition['midpoint'][0] - 20, transition['midpoint'][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
           
            source_idx = reordered_states.index(transition['source'])
            dest_idx = reordered_states.index(transition['destination'])
            transition_data.append({
                'source': int(source_idx),  # Convert to Python int
                'destination': int(dest_idx),  # Convert to Python int
                'label': str(label)  # Ensure label is a string
            })
        
      
        debug_path = 'static/fsa_analysis_result.png'
        cv2.imwrite(debug_path, result_image)
        print(f"Saved debug image to {debug_path}")
        cv2.imshow('image',result_image)
       
        fsa_data = {
            "states": state_info,
            "transitions": transition_data
        }
        
        return fsa_data
    except Exception as e:
        print(f"Error processing FSA image: {str(e)}")
        raise






class FSAChatbot:
    def __init__(self, fsa_data: Dict):
        self.fsa_data = fsa_data
        self.rules = self._create_rules()
        self.engine = pyttsx3.init() 
        self.recognizer = sr.Recognizer()  
        self.microphone = sr.Microphone()  
        self.engine.setProperty('rate', 120)
        
    def _create_rules(self) -> Dict:
        """
        Create predefined rules for the chatbot based on FSA data.
        """
        rules = {
            "states": self.fsa_data["states"],
            "transitions": self.fsa_data["transitions"],
            "initial_state": None,
            "final_state": None,
            "input_symbols": set()
        }
        
        
        for state in rules["states"]:
            if "Initial" in state["type"]:
                rules["initial_state"] = state
            if "Final" in state["type"]:
                rules["final_state"] = state
        
       
        for transition in rules["transitions"]:
            rules["input_symbols"].add(transition["label"])
        
        return rules

    def speak(self, text: str):
 
  
        self.engine.stop() 
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self) -> str:
   
      try:
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source) 
            self.speak("Please ask your question now")
            print("Listening...")
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
        question = self.recognizer.recognize_google(audio)
        print(f"You asked: {question}")
        return question.lower()
      except sr.WaitTimeoutError:
        self.speak("I didn't hear anything. Please try again.")
        return ""
      except sr.UnknownValueError:
        self.speak("Sorry, I couldn't understand your question. Please try again.")
        return ""
      except sr.RequestError:
        self.speak("Sorry, there was an error with the speech recognition service.")
        return ""
      except Exception as e:
        print(f"Error in listening: {str(e)}")
        return ""

    def answer_question(self, question: str) -> str:
        """
        Answer questions based on predefined rules and provide voice output.
        """
        if not question:
            return "No question detected."
            
        question = question.lower()
        
        # Rule 1: What are the states in the given FSA?
        if "states" in question and ("what" in question or "list" in question):
            def sort_key(state):
                if 'final' in state['type']:
                    return 0  # Final states come first
                elif 'normal' in state['type']:
                    return 1  # Normal states come second
                elif 'initial' in state['type']:
                    return 2  # Initial states come last
                else:
                    return 3  # Any other type (if exists)

            sorted_states = sorted(self.rules["states"], key=sort_key)
            states = [f"State {state['id']} ({', '.join(state['type'])})" 
                      for state in sorted_states]
            response = f"The states in the FSA are: {', '.join(states)}."
        
        # Rule 2: What are the transitions in the given FSA?
        elif "transitions" in question or "transitions" in question:  # Handling possible mispronunciation
            transitions = [f"State {t['source']} -> State {t['destination']} (Input: '{t['label']}')" 
                          for t in self.rules["transitions"]]
            response = f"The transitions in the FSA are: {', '.join(transitions)}."
        
        # Rule 3: What is the initial state?
        elif "initial state" in question or "initial" in question:
            if self.rules["initial_state"]:
                response = f"The initial state is State {self.rules['initial_state']['id']}."
            else:
                response = "No initial state detected."
        
        # Rule 4: What is the final state?
        elif "final state" in question or "final" in question:
            if self.rules["final_state"]:
                response = f"The final state is State {self.rules['final_state']['id']}."
            else:
                response = "No final state detected."
        
        # Rule 5: What is the input symbol for the transition from state X to state Y?
        elif ("input symbol" in question or "input" in question) and ("transition" in question or "from" in question):
            try:
                # Extract numbers from the question that might be state IDs
                numbers = [int(s) for s in question.split() if s.isdigit()]
                
                if len(numbers) >= 2:
                    source = numbers[0]
                    dest = numbers[1]
                    
                    # Search for the transition in the rules
                    for t in self.rules["transitions"]:
                        if t["source"] == source and t["destination"] == dest:
                            response = f"The input symbol for the transition from State {source} to State {dest} is '{t['label']}'."
                            break
                    else:
                        response = f"No transition found from State {source} to State {dest}."
                else:
                    response = "Please specify both the source and destination states in your question."
            except:
                response = "Sorry, I couldn't understand the states in your question. Please try asking again."
        
        # Rule 6: Basic conceptual questions about FSA
        elif "what is a finite state automata" in question or "what is an fsa" in question or "what is the finite state automata" in question:
            response = ("A Finite State Automaton (FSA) is a mathematical model of computation used to "
                        "design both computer programs and sequential logic circuits. It consists of a "
                        "finite number of states, transitions between these states, and actions.")
        
        elif "what is a state" in question:
            response = ("A state is a condition or situation of the FSA at a given time. It represents "
                        "a specific configuration of the system.")
        
        elif "what is a transition" in question:
            response = ("A transition is a change from one state to another in response to an input symbol.")
        
        elif "what is an input symbol" in question or "what are input symbols" in question:
            response = ("An input symbol is a character or token that triggers a transition between states "
                        "in an FSA.")
        
        
        else:
            response = "Sorry, I don't understand that question. Please ask about the FSA states, transitions, or basic concepts."
        
       
       
        return response

# ========================== Flask Routes ==========================

@app.route("/")
def home():
    return render_template("index.html")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    fsa_data = data.get("fsa_data")
    voice_input = data.get("voice_input", False)
    
    if not fsa_data:
        return jsonify({"error": "Invalid request - missing FSA data"}), 400
    
    print("Received FSA data:", fsa_data)
    
    try:
        
        chatbot = FSAChatbot(fsa_data)
        
     
        if voice_input:
            question = chatbot.listen()
            if not question:
                return jsonify({"response": "No question detected. Please try again."})
        
       
        if not question:
            return jsonify({"response": "No question detected. Please try again."})
        
      
        response = chatbot.answer_question(question)
        
        # Clean up resources
        del chatbot
        
        print("Chatbot response:", response)
        return jsonify({
            "response": response, 
            "question": question if question else "Voice question"
        })
    except Exception as e:
        print("Error in chatbot:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/process_image", methods=["POST"])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only PNG and JPG are allowed."}), 400
    
    
    upload_folder = "static"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    image_path = os.path.join(upload_folder, "uploaded_image.png")
    file.save(image_path)
    
   
    if not os.path.exists(image_path):
        return jsonify({"error": "Failed to save the uploaded image"}), 500
   
    try:
        fsa_data = process_fsa_image(image_path)
        return jsonify({"success": True, "fsa_data": fsa_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
