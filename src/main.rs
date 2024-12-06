use std::error::Error;
use opencv::highgui;
use opencv::prelude::*;
use tch::nn::ModuleT;
use tch::vision::imagenet::load_image;
use tch::{nn, nn::Module, Tensor};

// Weights and config:
// https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-spp.cfg
//

fn load_yolo_model() -> Result<Module, Box<dyn Error>> {
    // Load the pre-trained YOLO model
    let model = tch::CModule::load("yolov3.pt")?;
    Ok(Module::CModule(model))
}

fn detect_humans_in_frame(frame: &Mat, model: &Module) -> Result<Vec<Rect>, Box<dyn Error>> {
    // Convert the OpenCV frame to a PyTorch tensor
    let img = load_image(frame, false)?;
    let img_tensor = Tensor::of_slice(&img).permute(&[2, 0, 1]).unsqueeze(0).to_kind(tch::Kind::Float);

    // Run the YOLO model
    let detections = model.forward_is(&[img_tensor]).unwrap();

    // Post-process the detections and extract human body bounding boxes
    let detections = detections.squeeze_dim(0);
    let detections = detections.permute(&[1, 2, 0]);
    let detections = detections.view((detections.size()[0], -1, 85));
    let detections = detections.detach().to(tch::Kind::Float);

    let mut detected_humans = Vec::new();
    for i in 0..detections.size()[1] {
        let class_conf = detections.i((0, i, 4)).item::<f64>().unwrap();
        let class_id = detections.i((0, i, 5..85)).argmax(0, false).0.item::<i64>().unwrap();

        if class_conf > 0.5 && class_id == 0 {
            let bbox = detections.i((0, i, 0..4));
            let x1 = (frame.cols() as f64 * bbox.i((0, 0)).item::<f64>().unwrap()) as i32;
            let y1 = (frame.rows() as f64 * bbox.i((0, 1)).item::<f64>().unwrap()) as i32;
            let x2 = (frame.cols() as f64 * bbox.i((0, 2)).item::<f64>().unwrap()) as i32;
            let y2 = (frame.rows() as f64 * bbox.i((0, 3)).item::<f64>().unwrap()) as i32;
            detected_humans.push(Rect::new(x1, y1, x2 - x1, y2 - y1));
        }
    }

    Ok(detected_humans)
}

fn initialize() -> Result<(highgui::VideoCapture, Module), Box<dyn Error>> {
    // Load the YOLO model
    let model = load_yolo_model()?;

    // Open the default camera (index 0)
    let mut cap = highgui::VideoCapture::new(0)?;
    if !cap.is_opened()? {
        panic!("Failed to open the camera");
    }

    Ok((cap, model))
}

fn process_frame(frame: &Mat, model: &Module) -> Result<(), Box<dyn Error>> {
    let detected_humans = detect_humans_in_frame(&frame, &model)?;

    // Draw bounding boxes around detected humans
    for rect in detected_humans {
        highgui::rectangle(&frame, rect, Scalar::new(0., 255., 0., 0.), 2, 8, 0)?;
    }

    // Display the frame with bounding boxes
    highgui::imshow("Human Detection", &frame)?;
    let key = highgui::wait_key(1)?;
    if key == 27 {
        // Exit when the 'Esc' key is pressed
        return Err("Exiting".into());
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let (mut cap, model) = initialize()?;

    loop {
        // Read a frame from the camera
        let mut frame = Mat::default();
        cap.read(&mut frame)?;

        process_frame(&frame, &model)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    // Helper function to load the test video file.
    fn load_test_video(filename: &str) -> Result<highgui::VideoCapture, Box<dyn Error>> {
        let mut cap = highgui::VideoCapture::from_file(filename, highgui::VideoCaptureAPIs::CAP_FFMPEG)?;
        if !cap.is_opened()? {
            panic!("Failed to open the test video file");
        }
        Ok(cap)
    }

    #[rstest]
    #[case("tests/test_video_1.mp4", 3)] // Update the case with the correct expected number of detections.
    #[case("tests/test_video_2.mp4", 4)] // Update the case with the correct expected number of detections.
    // Add more test cases for different video files.
    fn test_human_detection(#[case] video_filename: &str, #[case] expected_detections: usize) {
        let (cap, model) = initialize().unwrap();
        let device = Device::CudaIfAvailable;
        let model = model.to(device);

        let mut frame = Mat::default();
        let mut frame_count = 0;
        let mut total_detected = 0;

        while cap.read(&mut frame).unwrap() {
            let detected_in_frame = count_detected_humans_in_frame(&frame, &model);
            total_detected += detected_in_frame;

            frame_count += 1;
            if frame_count == 100 {
                // Stop processing after 100 frames for quick test execution.
                break;
            }
        }

        assert_eq!(total_detected, expected_detections);
    }
}