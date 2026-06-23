//! Effector: drive a hand from a decoded gesture.
//!
//! The signal source can be a skin electrode or (later) an implant; the
//! effector doesn't care. `VirtualHand` closes the loop in-process; a physical
//! hand sits behind the same `Effector` trait later.

use tracing::info;

/// A commanded hand pose. For now just a named state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandPose {
    pub name: String,
}

impl HandPose {
    /// Map a decoded gesture label to a pose. Known grasps get short names;
    /// anything else passes through unchanged.
    pub fn from_class(label: &str) -> HandPose {
        let name = match label {
            "hand_open" => "open",
            "hand_close" => "close",
            other => other,
        };
        HandPose {
            name: name.to_string(),
        }
    }
}

/// Something that can be driven to a pose.
pub trait Effector {
    fn apply(&mut self, pose: &HandPose);
}

/// In-process virtual hand: tracks the current pose and logs transitions.
#[derive(Default)]
pub struct VirtualHand {
    current: Option<String>,
}

impl VirtualHand {
    pub fn new() -> Self {
        VirtualHand::default()
    }

    /// The current pose name, if any has been applied.
    pub fn current(&self) -> Option<&str> {
        self.current.as_deref()
    }
}

impl Effector for VirtualHand {
    fn apply(&mut self, pose: &HandPose) {
        if self.current.as_deref() != Some(pose.name.as_str()) {
            info!(from = ?self.current, to = %pose.name, "virtual hand pose change");
            self.current = Some(pose.name.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_maps_known_gestures_and_passes_through() {
        assert_eq!(HandPose::from_class("hand_open").name, "open");
        assert_eq!(HandPose::from_class("hand_close").name, "close");
        assert_eq!(HandPose::from_class("rest").name, "rest");
        // Unknown gesture labels pass through unchanged.
        assert_eq!(HandPose::from_class("wrist_flex").name, "wrist_flex");
    }

    #[test]
    fn virtual_hand_tracks_current_pose() {
        let mut hand = VirtualHand::new();
        assert_eq!(hand.current(), None);

        hand.apply(&HandPose::from_class("hand_open"));
        assert_eq!(hand.current(), Some("open"));

        hand.apply(&HandPose::from_class("hand_close"));
        assert_eq!(hand.current(), Some("close"));
    }

    #[test]
    fn applying_same_pose_keeps_it() {
        let mut hand = VirtualHand::new();
        hand.apply(&HandPose::from_class("rest"));
        hand.apply(&HandPose::from_class("rest"));
        assert_eq!(hand.current(), Some("rest"));
    }
}
