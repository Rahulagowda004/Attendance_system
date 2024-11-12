
# Automated Attendance System

The Automated Attendance System is a web application designed to streamline attendance processes in schools, universities, and other institutions. Using machine learning and computer vision techniques, this application can recognize faces from images, log attendance automatically, and reduce the time and effort needed for traditional attendance methods.

## Key Features

- Flexible Training Options: Train the model easily with multiple faces simultaneously or with a single face, making it adaptable to various environments.
- Real-Time Face Recognition: Uses YOLO and facenet for accurate face detection and recognition, ensuring reliable attendance logging.
- Capture-Based Attendance: Capture class attendance by simply taking photos, ideal for large groups where real-time recognition is needed.
- Web Interface: The system is deployed as a web application, making it accessible from any device with a browser.
- Scalable: Can be integrated into larger attendance management systems or used as a standalone solution.

## Technologies Used

- Languages & Frameworks: Python, HTML, Streamlit
- Libraries: facenet_pytorch, OpenCV, YOLO
- Other Tools: numpy, PIL, torch, and more

## Installation Instructions

To set up the Automated Attendance System locally, follow these steps:

1. Clone the Repository:

    ```bash
    git clone https://github.com/yourusername/automated-attendance-system.git
    cd automated-attendance-system
    ```

2. Set Up a Virtual Environment:

    ```bash
    python3.12 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install Dependencies: Install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Usage Instructions

1. Start the Web Application: Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

    This will start a local server, typically accessible at http://localhost:8501.

2. Configure Attendance Settings:
    - Upload or train the system with user images for face recognition.
    - Configure settings such as recognition thresholds, attendance logs, and session details.

3. Using the Attendance System:
    - Capture photos of the class, which the system will process to recognize faces and mark attendance automatically.
    - Recognized faces are marked as present in the attendance log, ready for review.

4. Accessing Attendance Logs:
    - Attendance logs are saved in a database or CSV file (customizable).
    - View or export attendance data through the application’s interface.

## Examples of Use

- School Environment: Capture a group photo of the classroom for easy attendance logging.
- Company Meetings: Track attendance by capturing an image of participants.

## Future Roadmap

- Mobile Compatibility: Develop a mobile-friendly version of the application.
- Enhanced Security: Add multi-factor authentication or other security features.
- Analytics Dashboard: Incorporate data analysis to track attendance trends and patterns.




## Known Issues / Bugs

- Camera Permissions: Ensure camera permissions are enabled on devices to allow video access.
- Face Recognition Limitations: Factors such as lighting, camera quality, and angle may impact recognition accuracy.

## Contributors

- Rahul A Gowda
- Santhosh R
- S Naveen Gowda
- Benhur Stephen Kumar 


