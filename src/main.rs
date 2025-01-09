use method::motion_detection::main::MotionDetection;
use opencv::{videoio::VideoCapture, Result};

mod method;

fn main() -> Result<()> {
    let video = VideoCapture::from_file(
        "/home/made/projects/spl3mpc/data_latih/video2.mkv",
        opencv::videoio::CAP_FFMPEG,
    )?;
    MotionDetection::create(video);

    Ok(())
}
