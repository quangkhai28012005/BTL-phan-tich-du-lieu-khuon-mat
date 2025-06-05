import cv2
import face_recognition
video_stream = cv2.VideoCapture('vids/team1.mov')
#truyền ảnh vào biến và encodigs từng ảnh mỗi thành viên
face_1 = face_recognition.load_image_file('images/khai.png')
face_1_encodings = face_recognition.face_encodings(face_1)[0]
face_1_name = 'Chu Khai'

face_3 = face_recognition.load_image_file('images/hien.jpg')
face_3_encodings = face_recognition.face_encodings(face_3)[0]
face_3_name = 'Phan Hien'

face_4 = face_recognition.load_image_file('images/khue.png')
face_4_encodings = face_recognition.face_encodings(face_4)[0]
face_4_name = 'Dinh Khue '

face_5 = face_recognition.load_image_file('images/huy.png')
face_5_encodings = face_recognition.face_encodings(face_5)[0]
face_5_name = 'Phan Huy'

face_6 = face_recognition.load_image_file('images/linh.png')
face_6_encodings = face_recognition.face_encodings(face_6)[0]
face_6_name = 'Thuy Linh'

#đặt tất cả encode vào 1 list để dùng vòng lặp
known_face_encodings = [face_3_encodings,face_1_encodings,face_4_encodings, face_5_encodings, face_6_encodings]
known_face_names = [face_3_name,face_1_name, face_4_name, face_5_name, face_6_name]

#tạo những biến để truyền tham số sau khi đọc được từ videos
all_face_locations = []
all_face_encodings = []
all_face_names = []

#vòng lặp vô hạn, rất lag :v
while True:
    ret, current_frame = video_stream.read()
    #cắt nhỏ khung hình đi 1/4 lần sao cho sát với mặt để tối ưu nhận diện
    scale_factor = 4
    current_frame_small = cv2.resize( current_frame, (0, 0), fx=1/scale_factor, fy=1/scale_factor)

    #tìm tất cả khuôn mặt có trong khung hình, encodings những mặt đó và để tên trống 
    all_face_locations = face_recognition.face_locations( current_frame_small, number_of_times_to_upsample=2, model='hog')
    #sử dụng model "cnn" thay vì "hog" nếu như webcam (vid) chất lượng thấp, môi trường quay không đảm bảo để cải thiện khả năng xử lí (tốn tài nguyên hơn "hog")
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)

    #lúc này dùng for lấy từng ảnh đã truyền từ trước để so sánh những khuôn mặt được encode trong hình
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        
        #đưa tọa độ về tương tự ảnh gốc
        top_pos *= scale_factor
        right_pos *= scale_factor
        bottom_pos *= scale_factor
        left_pos *= scale_factor
        
        #tách từng mặt từ khung hình ra
        current_face_image = current_frame[top_pos: bottom_pos, left_pos:right_pos]

        #so sánh với những mặt đã truyền vào với mặt được tách
        all_matches = face_recognition.compare_faces( known_face_encodings, current_face_encoding)

        #không thấy mặt tương thích sẽ trả về unknown
        name_of_person = 'Unknown Face'

        # Kiểm tra xem all_matches có trống không
        # Nếu có lấy số chỉ mục tương ứng với khuôn mặt trong chỉ mục đầu tiên
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]

        # vẽ khung quanh mặt, hiện tên 
        cv2.rectangle(current_frame, (left_pos, top_pos),
                      (right_pos, bottom_pos), (255, 255, 255), 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos,
                    bottom_pos + 20), font, 0.5, (0, 255, 0), 1)

        #hiện ảnh tìm thấy 
        cv2.imshow('Identified Faces', current_frame)
        
    # ấn nút q để out vòng lặp 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()
cv2.destroyAllWindows()
