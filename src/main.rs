use opencv::{
    core::{Mat, MatTraitConst},
    highgui,
    prelude::*,
    videoio::{VideoCapture, VideoCaptureTrait},
    Result,
};

fn main() -> Result<()> {
    // Muat berkas video
    let mut video = VideoCapture::from_file(
        "/home/made/projects/spl3mpc/data_latih/video2.mkv",
        opencv::videoio::CAP_FFMPEG,
    )?;
    if !video.is_opened()? {
        panic!("Video tidak bisa dimuat!");
    }

    loop {
        // Ambil tiap frame untuk ditampilkan
        let mut frame = Mat::default();
        video.read(&mut frame)?;
        if frame.empty() {
            break;
        }
        highgui::imshow("frame", &frame)?;

        // Tambah delay agar gambar dapat ditampilkan
        if highgui::wait_key(20)? >= 0 {
            break;
        }
    }

    Ok(())
}
