use opencv::{
    core::{Mat, MatTraitConst},
    highgui,
    prelude::*,
    videoio::{VideoCapture, VideoCaptureTrait},
    Result,
};

fn main() -> Result<()> {
    // Attempt to open the video file
    let mut video = VideoCapture::from_file(
        "/home/made/projects/spl3mpc/data_latih/video2.mkv",
        opencv::videoio::CAP_FFMPEG,
    )?; // Use `?` to propagate errors
    if !video.is_opened()? {
        panic!("Unable to open the video file!");
    }

    loop {
        let mut frame = Mat::default();
        // Read a frame from the video
        if !video.read(&mut frame)? || frame.empty() {
            break; // Break the loop if no more frames or frame is empty
        }

        // Display the frame in a window
        highgui::imshow("frame", &frame)?;

        // Add a delay to make the video visible
        if highgui::wait_key(30)? >= 0 {
            break; // Exit if a key is pressed
        }
    }

    Ok(())
}
