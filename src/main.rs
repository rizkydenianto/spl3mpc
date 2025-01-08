use opencv::{
    core::{self, Rect},
    highgui, imgproc,
    objdetect::CascadeClassifier,
    prelude::*,
    videoio,
};

fn main() -> opencv::Result<()> {
    let cascade_src = "/home/made/projects/spl3mpc/src/cars.xml";
    let video_src = "/home/made/projects/spl3mpc/data_latih/video.mp4";

    // Open the video file
    let mut cap = videoio::VideoCapture::from_file(video_src, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        panic!("Error opening video file!");
    }

    // Load the cascade classifier
    let mut car_cascade = CascadeClassifier::new(cascade_src)?;

    loop {
        let mut frame = Mat::default();
        if !cap.read(&mut frame)? || frame.empty() {
            break;
        }

        // Convert the frame to grayscale
        let mut gray = Mat::default();
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Detect cars in the frame
        let mut cars = core::Vector::<Rect>::new();
        car_cascade.detect_multi_scale(
            &gray,
            &mut cars,
            1.1,
            1,
            0,
            core::Size::new(30, 30),
            core::Size::default(),
        )?;

        // Draw rectangles around detected cars
        for car in cars {
            imgproc::rectangle(
                &mut frame,
                car,
                core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;
        }

        // Display the frame
        highgui::imshow("video", &frame)?;

        // Exit if the Esc key is pressed
        if highgui::wait_key(33)? == 27 {
            break;
        }
    }

    highgui::destroy_all_windows()?;
    Ok(())
}
