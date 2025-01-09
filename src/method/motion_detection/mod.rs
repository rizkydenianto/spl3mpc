use main::MotionDetection;
use opencv::{
    core::MatTraitConst,
    highgui,
    videoio::VideoCapture,
    Result,
};

pub mod main;

pub fn motion_detection_process(source: VideoCapture) -> Result<()> {
    let mut motion_detection = MotionDetection::create(source);

    loop {
        // Ambil citra
        motion_detection.take_frame()?;
        if motion_detection.frame.empty() {
            break;
        }

        // Buat citra abu-abu
        motion_detection.create_grayscale_frame();

        // Proses jika citra sebelumnya sudah ada
        if !motion_detection.prev_frame.empty() {
            // Implementasi pengaburan Gaussian
            let clone = motion_detection.gray_frame.clone();
            motion_detection.apply_gaussian_blur(clone)?;

            // Hitung perbedaan antara citra saat ini dan sebelumnya
            motion_detection.calculate_diff_frame();

            // Buat ambang batas untuk menyorot area yang terdapat pergerakan
            motion_detection.create_thresh_frame();

            // Implementasi pelebaran untuk mengisi celah
            let clone = motion_detection.thresh_frame.clone();
            motion_detection.apply_dilation(clone);

            // Tampilkan citra dengan area pergerakkan
            highgui::imshow("Movements", &motion_detection.thresh_frame)?;

            // Gambar persegi di area pergerakkan
            motion_detection.draw_rectangle();
        }

        // Tampilkan citra asli dnegan persegi
        highgui::imshow("Current Frame", &motion_detection.frame)?;

        // Update the previous frame
        motion_detection
            .gray_frame
            .copy_to(&mut motion_detection.prev_frame)?;

        // Tambah delay agar gambar dapat ditampilkan
        if highgui::wait_key(20)? >= 0 {
            break;
        }
    }

    Ok(())
}
