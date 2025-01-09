use opencv::{
    core::{self, Point, Rect, Scalar, Size},
    highgui, imgproc,
    objdetect::CascadeClassifier,
    prelude::*,
    videoio,
};

fn main() -> opencv::Result<()> {
    // Path to Haar cascade XML file (ensure it exists in your project directory)
    let cascade_path = "/home/made/projects/spl3mpc/src/cars.xml"; // Replace with the correct path
    let mut vehicle_cascade = CascadeClassifier::new(cascade_path)?;

    // Open the video file
    let video_path = "/home/made/projects/spl3mpc/data_latih/video.mp4"; // Replace with the correct video path
    let mut video_capture = videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY)?;
    if !video_capture.is_opened()? {
        panic!("Failed to open the video file: {}", video_path);
    }

    let mut frame = Mat::default();
    let mut vehicle_count = 0;
    let line_position = 300; // Y-coordinate for the counting line

    loop {
        // Read a frame from the video
        video_capture.read(&mut frame)?;
        if frame.empty() {
            break; // Exit when the video ends
        }

        // Convert frame to grayscale for Haar cascade
        let mut gray_frame = Mat::default();
        imgproc::cvt_color(&frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0)?;

        // Detect vehicles
        let mut vehicles = core::Vector::<core::Rect>::new();
        vehicle_cascade.detect_multi_scale(
            &gray_frame,
            &mut vehicles,
            1.1,  // Scale factor
            3,    // Min neighbors
            0,    // Flags
            Size::new(30, 30),  // Min size
            Size::new(300, 300), // Max size
        )?;

        // Process detected vehicles
        for vehicle in vehicles {
            // Draw a rectangle around each vehicle
            imgproc::rectangle(
                &mut frame,
                vehicle,
                Scalar::new(0.0, 255.0, 0.0, 0.0), // Green rectangle
                2,
                imgproc::LINE_8,
                0,
            )?;

            // Check if the vehicle crosses the counting line
            let center_y = vehicle.y + vehicle.height / 2;
            if (center_y - line_position).abs() <= 5 {
                vehicle_count += 1;
            }
        }

        // Draw the counting line
        let frame_cols = frame.cols();
        let _ = imgproc::line(
            &mut frame,
            Point::new(0, line_position),
            Point::new(frame_cols, line_position),
            Scalar::new(255.0, 0.0, 0.0, 0.0), // Red line
            2,
            imgproc::LINE_8,
            0,
        );

        // Display the vehicle count
        let text = format!("Vehicles: {}", vehicle_count);
        imgproc::put_text(
            &mut frame,
            &text,
            Point::new(50, 50),
            imgproc::FONT_HERSHEY_SIMPLEX,
            1.0,
            Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow text
            2,
            imgproc::LINE_8,
            false,
        )?;

        // Show the frame
        highgui::imshow("Vehicle Detection", &frame)?;
        if highgui::wait_key(30)? == 27 {
            break; // Exit when the ESC key is pressed
        }
    }
    Ok(())
}
