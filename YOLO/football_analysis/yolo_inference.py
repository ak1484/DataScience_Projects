from ultralytics import YOLO 

model = YOLO('models/best.pt')

results = model.predict('input_videos/5 FASTEST Premier League Players 2022_23.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)