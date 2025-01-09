use opencv::{
    core::{absdiff, Mat, MatExprTraitConst, Point, Scalar, Size, Vector, BORDER_DEFAULT, CV_8U},
    imgproc::{self, COLOR_BGR2GRAY},
    videoio::{VideoCapture, VideoCaptureTrait},
    Result,
};

pub struct MotionDetection {
    pub source: VideoCapture,
    pub frame: Mat,
    pub gray_frame: Mat,
    pub prev_frame: Mat,
    pub diff_frame: Mat,
    pub thresh_frame: Mat,
}

impl MotionDetection {
    pub fn create(source: VideoCapture) -> Self {
        Self {
            source,
            frame: Mat::default(),
            gray_frame: Mat::default(),
            prev_frame: Mat::default(),
            diff_frame: Mat::default(),
            thresh_frame: Mat::default(),
        }
    }

    pub fn take_frame(&mut self) -> Result<()> {
        self.source.read(&mut self.frame)?;
        Ok(())
    }

    pub fn create_grayscale_frame(&mut self) -> Result<()> {
        imgproc::cvt_color(&self.frame, &mut self.gray_frame, COLOR_BGR2GRAY, 0)?;
        Ok(())
    }

    pub fn apply_gaussian_blur(&mut self, clone: Mat) -> Result<()> {
        imgproc::gaussian_blur(
            &clone,
            &mut self.gray_frame,
            Size::new(5, 5),
            0.0,
            0.0,
            BORDER_DEFAULT,
        )?;
        Ok(())
    }

    pub fn calculate_diff_frame(&mut self) -> Result<()> {
        absdiff(&self.prev_frame, &self.gray_frame, &mut self.diff_frame)?;
        Ok(())
    }

    pub fn create_thresh_frame(&mut self) -> Result<()> {
        imgproc::threshold(
            &self.diff_frame,
            &mut self.thresh_frame,
            25.0,
            255.0,
            imgproc::THRESH_BINARY,
        )?;
        Ok(())
    }

    pub fn apply_dilation(&mut self, clone: Mat) -> Result<()> {
        let kernel = Mat::ones(3, 3, CV_8U)?.to_mat()?;
        imgproc::erode(
            &clone,
            &mut self.thresh_frame,
            &kernel,
            Point::new(-1, -1),
            1,
            0,
            Scalar::all(0.0),
        )?;
        imgproc::dilate(
            &clone,
            &mut self.thresh_frame,
            &kernel,
            Point::new(-1, -1),
            2,
            0,
            Scalar::all(0.0),
        )?;
        Ok(())
    }

    pub fn draw_rectangle(&mut self) -> Result<()> {
        let mut contours: Vector<Mat> = Vector::new();
        imgproc::find_contours(
            &self.thresh_frame,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        for contour in contours.iter() {
            let rect = imgproc::bounding_rect(&contour)?;
            imgproc::rectangle(
                &mut self.frame,
                rect,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;
        }
        Ok(())
    }
}
