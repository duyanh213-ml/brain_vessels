# Brain Vessels segmentation via Deep Learning

Trong thư mục này chứa toàn bộ các quy trình, từ việc tiền xử lý dữ liệu, huấn luyện mô hình, hậu xử lý và đánh giá cho dự án phân đoạn mạch máu não bằng học sâu, cụ thể là việc sử dụng mạng U-Net 2D

<br>

Cấu trúc của các thư mục được hiển thị như sau:


``` bash
vesselssegment
├── config
├── evaluation
├── loss_history
├── masks
├── model
├── model_save
├── patches
├── predict_series
├── processing_data
└── raw_data


```

Trong đó, nhiệm vụ cụ thể của các thư mục là:

- `raw_data`: Đây là thư mục có chứa 20 ca chụp cộng hưởng từ, được chia thành 3 tập `train`, `validation` và `test`, ở mỗi tập sẽ có các thư mục chứa các file ảnh dicom và nhãn tương ứng (segment). 20 ca chụp được chia với tỉ lệ train:dev:test là 12:4:4, trong đó 12 ca chụp dùng cho việc huấn luyện được đến từ 6 bệnh viện với mỗi bệnh viện 2 ca, để đảm bảo cho dữ liệu được cân bằng.

- `processing_data`: Thư mục này chứa toàn bộ các file Python có nhiệm vụ tiền xử lý dữ liệu. Dữ liệu được thu thập từ 20 ca chụp cộng hưởng từ (MRI), với các kiểu máy chụp xung TOF khác nhau. Với mỗi một ca chụp (một series), nó sẽ chứa các hình ảnh dicom 2D về hình ảnh não của bệnh nhân theo chiều chụp Axial (chiều theo hướng từ đỉnh đầu xuống vùng cổ). Tùy vào các loại máy chụp mà các file ảnh dicom này có thể có sự khác biệt về kích cỡ, số lượng, chỉ số WW, WL (để hiểu thêm về hai chỉ số này, có thể đọc thêm tại https://myctregistryreview.com/courses/my-ct-registry-review-demo/lessons/ct-physics/topic/window-width-and-window-level/). Do kích cỡ của ảnh khá lớn, nên trước hết, chúng sẽ được chia nhỏ thành các “patch”, Việc chia patch được lấy theo tỉ lệ 50:50 (50% số lượng đến từ các patch có trung tâm là điểm sáng trong nhãn, 50% cồn lại thì ngược lại),sau đó, dữ liệu sẽ được chuẩn hóa để cho cùng về một phân bố và cuối cùng được lưu lại trong các folder `patches` và `masks`

- `patches`, `masks`: Hai thư mục này được tạo từ việc tiền xử lý dữ liệu, được chia thành các mục tương ứng dùng cho việc training và testing. Có một điều chú ý về các thư mục con trong hai thư mục `patches` và `masks`, đó là có các thư mục con có đuôi `train_train` và `train_eval`. Các thư mục `train_train` là các thư mục mà các patches được sinh ra từ dữ liệu của thư mục `train` trong `raw_data`, điểm khác nhau là ở chỗ ở mục `train_train`, các patches được sinh từ việc lấy tỉ lệ 50:50 (Như đã nhắc ở trên), còn muc `train_eval` thì các patches được sinh theo “dạng lưới” để lấy ra toàn bộ các phần nhỏ của bức ảnh.


- `model`, `evaluation`: Thư mục này sẽ chứa các file Python xây dựng, huấn luyện và đánh giá mô hình. Mô hình được lựa chọn là mô hình Unet 2D với kích cỡ đầu vào của một patch là 96 x 96. Mô hình sau khi được huấn luyện sẽ được prunning và được lưu vào trong thư mục `model_save`. Việc huấn luyện sẽ sử dụng hàm `dice_coef_loss` để làm hàm tổn thất. Do bài toán của chúng ta số lượng ở hai loại nhãn (Số điểm được segment) sẽ khá lệch nên không phù hợp cho `Cross_entropy_loss` ở bài toán này. Các giá trị đánh giá model được lưu lại trong file evaluation.txt


- `config`: Thư mục này chứa các thông số, hằng số cho việc huấn luyện, đánh giá mô hình

- `loss_history`: Đây là thư mục để lưu lại các giá trị của hàm loss qua từng epoch


- `predict_series`: Đây là thư mục để dự đoán một ca hoàn chỉnh, sau khi chúng ta đã hoàn thiện model


Do lượng dữ liệu khá lớn, nên trong thư mục này ở các phần raw_data, sẽ không chứa các phần dữ liệu.
