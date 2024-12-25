# 🎓 Automated Attendance System  

The **Automated Attendance System** is an intelligent, web-based application designed to simplify and enhance attendance management processes. By leveraging advanced **machine learning** and **computer vision** techniques, it enables automatic face recognition and attendance logging, eliminating the need for traditional, time-consuming methods.

---

## 🚀 Key Features  

- **📸 Flexible Training Options**  
  Train the system with multiple or individual faces, making it adaptable to diverse environments.  

- **⚡ Real-Time Face Recognition**  
  Employs **YOLO** for face detection and **facenet** for high-accuracy face recognition, ensuring reliable attendance logging.  

- **📷 Capture-Based Attendance**  
  Allows attendance capture for large groups by simply uploading class photos, streamlining the process.  

- **🌐 Web-Based Interface**  
  Fully accessible via a user-friendly **Streamlit** web application.  

- **📊 Scalable Integration**  
  Can function as a standalone solution or integrate into larger attendance management systems.

---

## 🛠️ Technologies Used  

- **Programming Languages**: Python, HTML  
- **Frameworks**: Streamlit  
- **Libraries**:  
  - **Deep Learning**: facenet_pytorch, torch  
  - **Computer Vision**: OpenCV, YOLO  
  - **Utilities**: numpy, PIL  

---

## 📥 Installation Instructions  

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/yourusername/automated-attendance-system.git
cd automated-attendance-system
```

### 2️⃣ Set Up a Virtual Environment  

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies  

Install the required Python libraries:  

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage Instructions  

### 1️⃣ Start the Web Application  

Run the **Streamlit** application:  

```bash
streamlit run app.py
```

The local server will typically be available at: [http://localhost:8501](http://localhost:8501)  

---

### 2️⃣ Configure Attendance Settings  

- **Train the System**:  
  Upload user images or use a pre-trained model for face recognition.  

- **Adjust Settings**:  
  Configure parameters such as recognition thresholds, attendance logs, and session details.  

---

### 3️⃣ Capture and Log Attendance  

- **Capture Photos**: Upload a photo of the class or group.  
- **Automatic Recognition**: The system processes the image to recognize faces and logs attendance automatically.  
- **Attendance Logs**: Recognized faces are marked as "present," and attendance data is stored for easy access.  

---

### 4️⃣ Export Attendance Logs  

- Attendance data is saved as a **CSV file** or in a **database** (based on configuration).  
- Logs can be viewed or exported through the application interface.  

---

## 📊 Examples of Use  

- **📚 Educational Institutions**:  
  Automate attendance by capturing a classroom photo.  

- **🏢 Corporate Meetings**:  
  Record attendance by taking a single snapshot of participants.  

---

## 🔮 Future Roadmap  

- **📱 Mobile Compatibility**:  
  Develop a mobile-friendly version for ease of use on smartphones.  

- **🔒 Enhanced Security**:  
  Implement multi-factor authentication and advanced encryption techniques.  

- **📈 Analytics Dashboard**:  
  Add features for tracking attendance trends and generating insightful reports.  

---

## 👥 Contributors  

- **Rahul A Gowda**  
- **Santhosh R**  
- **S Naveen Gowda**  
- **Benhur Stephen Kumar**  

---

## 📜 License  

This project is licensed under the **MIT License**. For more details, see the [LICENSE](LICENSE) file.  

---

This version enhances readability and provides a professional, well-organized structure, making it appealing for users and contributors alike.
