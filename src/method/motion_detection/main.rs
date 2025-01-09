use opencv::videoio::VideoCapture;

pub struct MotionDetection {
    source: VideoCapture,
}

impl MotionDetection {
    pub fn create(source: VideoCapture) -> Self {
        Self { source }
    }
}
