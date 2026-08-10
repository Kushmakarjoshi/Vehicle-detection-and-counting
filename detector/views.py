import os
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from PIL import Image

DEFAULT_MODEL_PATH = os.path.join(settings.BASE_DIR, "models", "best_old.pt")
VEHICLE_KEYWORDS = ("car", "truck", "bus", "van", "motor", "motorcycle", "bicycle", "vehicle")


def count_vehicles(detections):
    count = 0
    for detection in detections:
        class_name = str(detection.get("class_name", "") or "").strip().lower()
        if not class_name:
            count += 1
            continue
        if any(keyword in class_name for keyword in VEHICLE_KEYWORDS):
            count += 1
    return count


def home(request):
    model_name = os.path.basename(DEFAULT_MODEL_PATH) if os.path.exists(DEFAULT_MODEL_PATH) else "No default model found"
    return render(request, "home.html", {"model_name": model_name})


def upload_image(request):
    if request.method != "POST":
        return HttpResponseRedirect(reverse("home"))

    model_file = request.FILES.get("model_file")
    image_file = request.FILES.get("image_file")

    if not image_file:
        return render(request, "home.html", {"error": "Please upload an image.", "model_name": os.path.basename(DEFAULT_MODEL_PATH)})

    if model_file:
        model_dir = os.path.join(settings.MEDIA_ROOT, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_file.name)
        with open(model_path, "wb+") as destination:
            for chunk in model_file.chunks():
                destination.write(chunk)
    else:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            return render(request, "home.html", {"error": "No default model was found in the models folder.", "model_name": "No model found"})
        model_path = DEFAULT_MODEL_PATH

    image_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(image_dir, exist_ok=True)
    image_name = os.path.basename(image_file.name)
    image_path = os.path.join(image_dir, image_name)
    with open(image_path, "wb+") as destination:
        for chunk in image_file.chunks():
            destination.write(chunk)

    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        results = model(image_path, conf=0.25, stream=False)

        result_image_path = None
        detections = []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", None)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = None
                if isinstance(names, dict):
                    class_name = names.get(cls)
                elif isinstance(names, (list, tuple)) and cls < len(names):
                    class_name = names[cls]
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": round(conf, 2),
                    "class_id": cls,
                    "class_name": class_name,
                })

        if hasattr(result, "plot"):
            image_array = result.plot()
            if image_array is not None:
                img = Image.fromarray(image_array)
                output_path = os.path.join(settings.MEDIA_ROOT, "results", image_name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img.save(output_path)
                result_image_path = output_path

        vehicle_count = count_vehicles(detections)

        return render(request, "result.html", {
            "image_url": "/media/uploads/" + image_name,
            "result_image_url": "/media/results/" + image_name if result_image_path else None,
            "detections": detections,
            "vehicle_count": vehicle_count,
        })
    except ImportError as exc:
        return render(request, "home.html", {"error": f"YOLO dependencies are missing: {exc}", "model_name": os.path.basename(DEFAULT_MODEL_PATH)})
    except Exception as exc:
        return render(request, "home.html", {"error": f"Model run failed: {exc}", "model_name": os.path.basename(DEFAULT_MODEL_PATH)})
