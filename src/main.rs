use opencv::{core::{Mat, MatTraitConst}, highgui, videoio::{VideoCapture, VideoCaptureTrait, CAP_ANY}, Result};

fn main() -> Result<()> {
    let mut video = VideoCapture::from_file("video.mkv", CAP_ANY);
    if !video.is_ok() {
        panic!("Unable to open the video file!");
    }

    loop {
        let mut frame = Mat::default();
        video.as_mut().unwrap().read(&mut frame).unwrap();
        if frame.empty() {break}

        highgui::imshow("frame", &frame).unwrap();
    }

	Ok(())
}