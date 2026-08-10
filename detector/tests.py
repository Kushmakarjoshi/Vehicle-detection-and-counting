from django.test import SimpleTestCase

from detector.views import count_vehicles


class VehicleCountTests(SimpleTestCase):
    def test_counts_vehicle_like_labels(self):
        detections = [
            {"class_name": "car"},
            {"class_name": "truck"},
            {"class_name": "person"},
            {"class_name": "bus"},
        ]

        self.assertEqual(count_vehicles(detections), 3)

    def test_counts_all_when_labels_are_missing(self):
        detections = [{"class_name": None}, {"class_name": ""}]

        self.assertEqual(count_vehicles(detections), 2)
