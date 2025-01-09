use method::motion_detection::motion_detection_process;
use opencv::{
    prelude::*,
    videoio::VideoCapture,
    Result,
};

mod method;

fn main() -> Result<()> {
    // Muat berkas video
    let video = VideoCapture::from_file(
        "/home/wsl/Proyek/spl3mpc/data_latih/2025-01-08 13-22-42.mkv",
        opencv::videoio::CAP_FFMPEG,
    )?;
    if !video.is_opened()? {
        panic!("Video tidak bisa dimuat!");
    }

    motion_detection_process(video)?;
    Ok(())
}
