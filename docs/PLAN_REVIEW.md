# Review khung thực nghiệm 7 bảng — cập nhật theo kết quả đo được

*Viết lại sau khi có kết quả fine-tune weather. Mọi đánh giá dưới đây neo vào
số đã đo, không phải phỏng đoán.*

---

## 1. Điều gì đã thay đổi kể từ lúc viết plan

Plan 7 bảng được soạn khi chưa có bằng chứng nào về trục robustness. Giờ đã có,
và nó mạnh hơn dự kiến:

| | Model gốc | Sau fine-tune weather |
|---|---:|---:|
| `clean` AP | 0.1714 | **0.1750** *(không mất gì)* |
| `rain_heavy` AP | 0.0295 | **0.1246** — **4.22×** |
| `rain_heavy` giữ được | 17.2 % | **71.2 %** |
| `blur_medium` giữ được | 73.2 % | **84.3 %** |
| Số điều kiện cải thiện | — | **10/10** |

Con số quan trọng nhất không nằm trong bảng trên:

> **Cải thiện trung bình trên điều kiện *held-out* là 1.70×, cao hơn trên điều
> kiện *đã train* (1.27×).**

Model học được thứ tổng quát về ảnh suy biến chứ không thuộc lòng bốn điều kiện
được dạy. `blur_medium` tăng 1.18× dù chỉ được học `blur_light`; `fog_medium`
tăng dù chưa từng thấy sương. Đây là loại bằng chứng khó bác.

**Hệ quả cho plan:** trục robustness không còn là "thứ lấp chỗ trống". Nó là
trục có kết quả hoàn chỉnh, có cặp before/after trên cùng giao thức, và có tính
tổng quát hoá đo được. Bảy bảng trong plan hiện tại **không có một ô nào** về
nó.

---

## 2. Vấn đề chặn: 5 trong 7 bảng cần thiết bị chưa vào được

Board chưa kết nối. Chẩn đoán: cổng Type-C trên box nhiều khả năng là cổng
nguồn — đèn sáng nhưng Windows không thấy tín hiệu USB nào qua mọi lần quét,
và mục "thiết bị lỗi/thiếu driver" trống rỗng ở tất cả các lần. Board có 4 cổng
RJ45 đang trống. **Thiếu đúng một sợi cáp mạng.**

Nhưng phần lớn plan vẫn cứu được, vì **AI Hub Workbench chạy job trên thiết bị
thật trên cloud**:

| Bảng | Cần gì | Khả thi không |
|---|---|---|
| 3 — thang trung thực | compile + quantize + inference job | ✅ chỉ cần AI Hub |
| 4 — lược đồ lượng tử | quantize + profile job | ✅ chỉ cần AI Hub |
| 6 — phân rã theo khối | profile job per-layer | ✅ chỉ cần AI Hub |
| 7 — NMS-free & fallback | **1 compile job** | ✅ rẻ nhất, đòn bẩy lớn nhất |
| 5 — ngân sách frame | capture + H2D/D2H + tracker | ❌ **bắt buộc board thật** |

Bảng 5 — bảng mình đánh giá cao nhất trong plan — lại là bảng duy nhất bị chặn
cứng.

---

## 3. Một bài học mới, phải đưa vào "quy ước bắt buộc"

Lần chạy benchmark model weather cho thấy điều này rõ ràng:

> `infer_ms` là forward pass cố định — cùng model, cùng 640×640, **khối lượng
> tính toán y hệt nhau bất kể ảnh chứa gì**. Nhưng nó dao động **22 %** giữa các
> điều kiện (32.2 → 39.2 ms). `fog_medium` chạy mất 153 giây so với trung vị 86
> giây.

Đó là tải máy và trạng thái nhiệt, không phải dữ liệu. Và cùng loại nhiễu đó
làm hỏng luôn các cột `post_ms` đo cạnh nó.

**Quy ước cần thêm:** *mọi bảng latency phải kèm một đại lượng bất biến để kiểm
chứng.* Ở đây `infer_ms` đóng vai trò đó — nếu nó lệch quá ~5 % giữa các điều
kiện, cả bảng latency của lần chạy đó phải bị loại. `report.py` giờ tự phát hiện
và in cảnh báo.

Hệ quả trực tiếp: **mọi số latency đo trên laptop đều không dùng được cho báo
cáo.** Chỉ có số đo trên board mới có nghĩa — điều này càng làm sợi cáp RJ45
thành việc ưu tiên số một.

---

## 4. Duyệt từng bảng

### Giữ nguyên

**Table 1 — Cấu hình thực nghiệm.** Đặt đầu Section 4 là đúng. Repo đã ghi sẵn
`model_sha256` + `backend` + `conf` + `IoU` + `maxDets` + `ignore_policy` vào
từng dòng kết quả.

*Một chỉnh sửa:* nhóm đã chỉ ra `maxDets` phải là **500** theo giao thức
VisDrone chứ không phải 300. Đúng, và đã sửa. Đã đo tác động trước khi đổi:
AP 0.1864 → 0.1872 (số detection/ảnh 231 → 327). Chênh **+0.0008**, không đổi
kết luận nào — nhưng giá trị của việc khai báo không nằm ở con số mà ở tính so
sánh được.

**Table 7 — NMS-free & fallback đồ thị.** *Ưu tiên số 1.* Một job compile,
**không cần train** (ONNX random-init cho kết quả fallback y hệt ONNX đã train).
Nếu YOLO26 e2e rơi fallback trên HTP thì đó là phát hiện chưa ai công bố cho
Hexagon. Rẻ nhất, đòn bẩy lớn nhất.

*Cảnh báo:* chưa ai trong nhóm kiểm chứng YOLO26 export QNN được hay không.
Đừng xây kế hoạch trên giả định đó — để một job compile trả lời trước.

**Table 3 — Thang trung thực qua chuỗi chuyển đổi.** Ý hay nhất về mặt phương
pháp trong cả plan. Cosine similarity S4 vs S1 tách được lỗi lượng tử hoá khỏi
lỗi compiler, và AI Hub inference job trả sẵn số đó. Ít người làm.

**Table 5 — Ngân sách một frame.** Dòng `NPU utilisation (%)` là dòng mạnh nhất
cả bài: nếu NPU chỉ bận 35 % ngân sách frame thì mọi tối ưu lượng tử hoá tiếp
theo đều vô nghĩa — và đó là một kết luận. Giữ, nhưng chấp nhận nó chặn ở board.

### Thu hẹp

**Table 2 + 4 — 4 model × 5 lược đồ.** Tách theo trục:

- **Trục latency / fallback (Table 2, 7): giữ cả 4 model.** Không cần train,
  ~16 job AI Hub, 0 giờ GPU.
- **Trục chất lượng (Table 3, 4): chỉ model của nhóm.** Ba model còn lại phải
  fine-tune VisDrone = 6–9 giờ GPU + 4 lượt benchmark. Và nó mâu thuẫn trực tiếp
  với lời khuyên "1 model" đã nhận từ đầu.

5 lược đồ × 1 model đã trả lời trọn vẹn câu hỏi triển khai. 5 × 4 = 20 ô là chi
phí không mua thêm được kết luận nào.

### Xuống phụ lục

**Table 6 — Phân rã theo khối kiến trúc.** Trùng Table 5 ở mức thô hơn. Caveat
nhóm tự nêu — *"instrumentation làm chậm chính nó"* — là đúng và tinh, nhưng nó
cũng chính là lý do bảng này khó bảo vệ. Để phụ lục.

### Bỏ hẳn

**Tạo nhiều tài khoản AI Hub.** Vi phạm điều khoản dịch vụ. Rủi ro thực tế: bị
khoá giữa chừng là mất sạch job, artefact và cả đường compile — đúng lúc gần
deadline. Và **không cần thiết**: job AI Hub chạy bất đồng bộ, submit 20 job rồi
đi làm việc khác. Nút thắt là thời gian chờ queue, mà nhiều tài khoản cũng không
rút ngắn được queue.

**Benchmark nhiều thuật toán tracking.** Không có ground-truth MOT thì chỉ đo
được latency — và như vậy tracker nặng **luôn luôn thua**, kết luận vô nghĩa.
Muốn nói "hiệu quả hơn" phải có VisDrone-MOT + TrackEval + IDF1/HOTA: chuyển GT
sang MOTChallenge, xử lý ignored region cho tracking — cả một nhánh đã bị cắt từ
đầu vì đúng lý do đó.

*Cách đúng và rẻ:* báo cáo `tracker_ms` như một dòng trong Table 5, không claim
gì về độ chính xác.

**Mann–Whitney U.** Hơi thừa. Nếu A nhanh gấp đôi B thì không cần kiểm định.
Đề nghị: mean ± std qua 3 lần chạy, chỉ dùng kiểm định khi chênh lệch dưới ~10 %.

---

## 5. Thứ plan đang thiếu

### 5.1. Trục robustness biến mất hoàn toàn

Bảy bảng không có ô nào về mưa/tối/mờ. Nếu theo đúng plan này, bài trở thành
**một nghiên cứu triển khai lượng tử hoá** — chủ đề đông người làm và nhóm không
có lợi thế gì. Còn bám robustness thì có góc riêng, và giờ đã có kết quả đầy đủ.

**Đề nghị: robustness là xương sống, lượng tử hoá là trục hai.**

### 5.2. Ô đáng giá nhất nằm ở chỗ hai trục giao nhau

> **Lượng tử hoá INT8 có làm model dễ vỡ hơn dưới mưa không?**

Cả hai đều nén dải động. Giả thuyết: `W8A8` mất AP nhiều hơn hẳn trên ảnh suy
biến so với ảnh sạch. Nếu đúng, kết luận rất mạnh cho người triển khai: *bảng
lượng tử hoá đo trên ảnh sạch đã đánh giá quá lạc quan chi phí thật khi bay
ngoài trời.*

Chưa ai công bố ô này, và nó chỉ tốn thêm việc chạy `bench_quality --all-conditions`
lên model đã quantize — thứ dù sao cũng phải làm.

| Scheme | AP sạch | AP mưa vừa | Δ do mưa |
|---|---|---|---|
| FP32 | 0.1750 | 0.1563 | −11 % |
| W8A8 | ? | ? | **> −11 % ?** |
| W8A16 | ? | ? | ? |

Đây nên là **bảng chính**.

### 5.3. Không có cột năng lượng

"Hiệu năng trên watt" là toàn bộ lý do NPU tồn tại, mà plan không có ô nào về
nó. `4-bench/probe_power.py` đã sẵn sàng: nó dò xem board lộ ra rail điện nào,
và **báo "không có" thay vì báo 0** nếu board không đo được. Nếu không có
counter năng lượng, dùng nhiệt độ + tần số làm proxy và ghi rõ.

### 5.4. Phát hiện "mù nhưng latency đẹp" chưa nằm trong bảng nào

Ở ngưỡng triển khai (`conf 0.25`), mưa nặng làm pipeline **nhanh hơn** vì nó
nhìn thấy ít hơn — detection 64.7 → 30.1/frame, NMS 3.2 → 1.8 ms. Dashboard
nhìn FPS/latency sẽ thấy "xanh" đúng lúc drone mù, và vì `vehicle_count` cũng
tụt nên hệ thống báo "giao thông bình thường" trong khi sự thật là "không nhìn
thấy gì".

Đây là kết luận đắt nhất của cả dự án và nó không thuộc bảng nào trong plan.

### 5.5. Bối cảnh cho con số AP tuyệt đối

AP 0.175 nghe thấp, nhưng phải trình bày kèm bối cảnh nếu không người đọc sẽ
hiểu sai:

- AP lấy trung bình qua IoU 0.50–0.95 và chia **đều 10 lớp**, kể cả lớp hiếm và
  cực khó (`bicycle` AP 0.0254, `awning-tricycle` 0.0517).
- Bốn lớp khó nhất chiếm 20.6 % số object nhưng kéo trung bình xuống ~40 %.
- Tính **trọng số theo số object thật**: AP = **0.259**.
- Riêng **nhóm phương tiện** (car/van/truck/bus, 17 040 object): AP = **0.442**.
- `car` riêng lẻ: **0.4916**.

Demo giao thông chủ yếu hiển thị xe, nên chất lượng người xem cảm nhận gần với
0.44 hơn là 0.175.

---

## 6. Khung đề xuất

Sáu bảng thay vì bảy, sắp lại theo trục:

| # | Bảng | Trạng thái | Chặn bởi |
|---|---|---|---|
| **1** | Cấu hình thực nghiệm | ✅ có sẵn | — |
| **2** | **Robustness: before / after fine-tune** ★★ | ✅ **đã xong** | — |
| **3** | Thang trung thực FP32 → ctx-bin | ⬜ | AI Hub token |
| **4** | **Lượng tử hoá × suy biến** ★★ | ⬜ | AI Hub token |
| **5** | Ngân sách frame NPU vs CPU | ⬜ | **cáp RJ45** |
| **6** | NMS-free & fallback đồ thị ★ | ⬜ | AI Hub token |

Bảng 2 đã hoàn chỉnh và là bảng mạnh nhất đang có. Bảng 4 là bảng mới, và là
đóng góp thật sự mới của bài.

**Ba hình, không hơn:**
1. AP theo 10 điều kiện, before vs after *(dữ liệu đã có)*
2. Pareto AP–latency qua các lược đồ lượng tử *(cần AI Hub)*
3. Stacked bar ngân sách frame *(cần board)*

---

## 7. Thứ tự thực hiện

| Ưu tiên | Việc | Phụ thuộc | Chi phí |
|---|---|---|---|
| 1 | **Kiếm 1 sợi cáp RJ45** | — | 0 |
| 2 | Job compile YOLO26 e2e → trả lời câu NMS-free | token | 1 job |
| 3 | `make_calibration.py` + `aihub_jobs.py` | token | ~1 h code |
| 4 | Quantize → chạy `bench_quality --all-conditions` → **Bảng 4** | #3 | ~1 h queue + 40 ph eval |
| 5 | Ngân sách frame trên board → **Bảng 5** | #1 | — |
| 6 | Quay demo (pipeline + dashboard) | — | ~1 h |

Việc #1 rẻ nhất và mở khoá nhiều nhất. Việc #6 làm được ngay, không phụ thuộc gì.

---

## 8. Quy ước — giữ nguyên của nhóm, bổ sung hai điều

Phần "quy ước bắt buộc" nhóm viết **rất tốt** và repo đã thực thi sẵn. Bổ sung:

1. **Mọi bảng latency phải kèm một đại lượng bất biến để kiểm chứng.** `infer_ms`
   không được phép đổi theo nội dung ảnh; nếu nó lệch >5 % thì loại cả bảng
   latency của lần chạy đó.
2. **Mọi phát biểu về chi phí latency của suy biến phải ghi rõ ngưỡng confidence
   *và* model đã đo trên.** Dấu của hiệu ứng đảo chiều giữa `conf 0.001` và
   `conf 0.25`.

---

## 9. Ba điều nên nói và không nên nói

**Nên nói:**

> Chúng em đánh giá nghiêm ngặt Task 1 — 548 ảnh, 10 điều kiện suy biến, giao
> thức đầy đủ. Tracking và đếm được **trình diễn** bằng cùng pipeline, nhưng
> chúng em **không báo cáo số** cho Task 2/4/5 vì chưa đánh giá theo đúng metric
> của chúng. Đổi lại chúng em bổ sung một trục không task nào trong bốn task đó
> có: độ bền dưới điều kiện thu ảnh suy biến, đo được và sửa được.

**Không nên nói:**

- *"Model chịu được mưa thật"* — mưa ở đây là **tổng hợp**. Không có giọt nước
  bám ống kính, không có phản xạ mặt đường ướt, không có nhiễu cảm biến trong
  ảnh tối. Đây là phép thử stress có kiểm soát, tái lập được — không phải bằng
  chứng drone bay được ngoài trời mưa.
- Bất kỳ con số latency nào đo trên laptop.
- *"First to use YOLO26"* hoặc bất kỳ claim novelty nào ở YOLO11n.
- Bất kỳ số nào về tracking accuracy (IDF1, IDSW, MOTA) — chưa hề đo.

---

## 10. Kết luận

Plan 7 bảng được viết tốt về mặt kỹ thuật đo đạc, và phần "quy ước bắt buộc"
cho thấy nhóm hiểu vấn đề tái lập. Ba điều cần đổi:

1. **Đưa robustness trở lại làm xương sống** — nó đã có kết quả hoàn chỉnh, có
   cặp before/after, và có tính tổng quát hoá đo được. Bỏ nó đi là bỏ đúng thứ
   phân biệt bài này với hàng trăm bài benchmark lượng tử hoá khác.
2. **Thu 4 model × 5 lược đồ về 1 model × 5 lược đồ cho trục chất lượng**, giữ
   4 model cho trục latency/fallback vốn không tốn giờ GPU nào.
3. **Thêm ô giao nhau giữa hai trục** — lượng tử hoá × suy biến. Đó là ô chưa ai
   công bố, và nó gần như miễn phí khi đã có cả hai trục.

Và bỏ ý tạo nhiều tài khoản AI Hub.
