Thành viên :
-	Phạm Xuân Hoàng - 20520519
----------------
Training : 
-	Việc training vui lòng truy cập link dataset kaggle : https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/
-	Import file training.ipynb lên kaggle, chọn GPU
-	Nhấn Run All để huấn luyện.
----------------
Inference : 
	-	Cài đặt các thư viện trong source/requirements.txt
	-	Download model : https://drive.google.com/drive/folders/1Tkp8aJ1F_C8gW3IZEW1Vhyi7wXQ3BFFF?usp=sharing
	-	Chạy file app.py bằng lệnh python app.py, hiện giao diện app. 
	-	Chọn đường dẫn tới model vừa download
	+	Nếu muốn dùng ảnh hoặc video, chọn đường dẫn tới ảnh hoặc video chứa mặt người vào trường đầu tiên, nhấn "Run Script".
	+	Nếu muôn dùng camera, chỉ cần nhấn "Run Live Cam"
	-	Nếu không muốn dùng app.py xin làm theo bước sau :
	+	Với ảnh : python inference.py --model_path path/to/your/model.h5 --img_path path/to/your/image.jpg
	+	Với video  : python inference.py --model_path path/to/your/model.h5 --video_path path/to/your/image.mp4 \
	+	Với camera : python inference.py --model_path path/to/your/model.h5
Lưu ý : 
-	Nếu gặp các vấn đề về việc đường dẫn model trong quá trình inference, copy model vào thư mục source/models và đổi tên model thành model.py
-	Hiện chỉ hỗ trợ các định dạng file ảnh .jpg và .png, file video .mp4 và model .h5