#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from classifier import classifier

def main():
    test_image = "pet_images/download (1).jpeg"
    model = "vgg"
    image_classification = classifier(test_image, model)
    print("\nResults from test_classifier.py\nImage:", test_image, "using model:",
          model, "was classified as a:", image_classification)

if __name__ == "__main__":
    main()
