use cgmath::prelude::*;
use cgmath::vec3;
use cgmath::Matrix3;
use cgmath::Vector3;
use eframe::egui;
use egui::NumExt;
use itertools::Itertools;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Blocks",
        native_options,
        Box::new(|cc| Box::new(App::new(cc))),
    )
}

struct App {
    world: World,
    frame_time: std::time::Instant,
    view: Viewport,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        Self {
            world: World {
                blocks: vec![Wireframe::block(vec3(-1., -1., 1.), vec3(1., 1., 1.))],
                camera: Camera {
                    pos: vec3(0., 0., 0.),
                    pitch: cgmath::Rad::zero(),
                    yaw: cgmath::Rad::zero(),
                },
            },
            frame_time: std::time::Instant::now(),
            view: Viewport {
                rect: cc.egui_ctx.screen_rect(),
            },
        }
    }

    fn draw_world(&self, ui: &mut egui::Ui) {
        let to_cam = |&p| self.world.camera.world_to_camera(p);
        let to_screen = |p| {
            self.view
                .camera_to_screen(self.world.camera.camera_to_clip(p))
        };

        for block in &self.world.blocks {
            let verts = block.verts.iter().map(to_cam).collect_vec();
            for &[i, j] in &block.edges {
                let mut a = verts[i];
                let mut b = verts[j];
                if b.z < a.z {
                    std::mem::swap(&mut a, &mut b);
                }
                if b.z < Camera::NEAR_PLANE {
                    continue;
                }
                if a.z < Camera::NEAR_PLANE {
                    let da = a.z - Camera::NEAR_PLANE;
                    let db = b.z - Camera::NEAR_PLANE;
                    a = (db * a - da * b) / (db - da);
                }
                ui.painter()
                    .line_segment([to_screen(a), to_screen(b)], (2.0, egui::Color32::GREEN))
            }
        }
    }
}

#[derive(Debug, Clone)]
struct World {
    blocks: Vec<Wireframe>,
    camera: Camera,
}
impl World {}

#[derive(Debug, Clone)]
struct Wireframe {
    verts: Vec<Vector3<f32>>,
    edges: Vec<[usize; 2]>,
}
impl Wireframe {
    fn block(min: Vector3<f32>, size: Vector3<f32>) -> Self {
        let a = min;
        let b = min + size;
        let verts = vec![
            vec3(a.x, a.y, a.z),
            vec3(b.x, a.y, a.z),
            vec3(a.x, b.y, a.z),
            vec3(b.x, b.y, a.z),
            vec3(a.x, a.y, b.z),
            vec3(b.x, a.y, b.z),
            vec3(a.x, b.y, b.z),
            vec3(b.x, b.y, b.z),
        ];
        let edges = vec![
            [0b000, 0b001],
            [0b000, 0b010],
            [0b000, 0b100],
            [0b001, 0b011],
            [0b001, 0b101],
            [0b010, 0b011],
            [0b010, 0b110],
            [0b100, 0b101],
            [0b100, 0b110],
            [0b011, 0b111],
            [0b011, 0b111],
            [0b101, 0b111],
            [0b101, 0b111],
            [0b110, 0b111],
            [0b110, 0b111],
        ];
        Self { verts, edges }
    }
}

#[derive(Debug, Clone)]
struct Camera {
    // in world space
    pos: Vector3<f32>,
    pitch: cgmath::Rad<f32>,
    yaw: cgmath::Rad<f32>,
}
impl Camera {
    const NEAR_PLANE: f32 = 0.001;

    fn rot_mat(&self) -> Matrix3<f32> {
        Matrix3::from(cgmath::Euler::new(
            self.pitch,
            self.yaw,
            cgmath::Rad::zero(),
        ))
    }
    fn yaw_mat(&self) -> Matrix3<f32> {
        Matrix3::from(cgmath::Euler::new(
            cgmath::Rad::zero(),
            self.yaw,
            cgmath::Rad::zero(),
        ))
    }
    fn world_to_camera(&self, world_pos: Vector3<f32>) -> Vector3<f32> {
        self.rot_mat() * (world_pos - self.pos)
    }

    fn camera_to_clip(&self, cam_pos: Vector3<f32>) -> egui::Vec2 {
        egui::vec2(cam_pos.x / cam_pos.z, cam_pos.y / cam_pos.z)
    }
}

struct Viewport {
    rect: egui::Rect,
}
impl Viewport {
    fn camera_to_screen(&self, cam_vec: egui::Vec2) -> egui::Pos2 {
        self.rect.center() + self.rect.size().min_elem() * cam_vec * egui::vec2(1., -1.) / 2.
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let last_frame = self.frame_time;
        self.frame_time = std::time::Instant::now();
        let dt = (self.frame_time - last_frame)
            .as_secs_f32()
            .at_most(1. / 30.);
        let move_speed = 1.;
        let rot_speed = cgmath::Rad::full_turn() / 6.;

        let mut dpos = vec3(0., 0., 0.);
        let mut dpitch = cgmath::Rad::zero();
        let mut dyaw = cgmath::Rad::zero();

        fn pressed(ctx: &egui::Context, key: egui::Key) -> bool {
            ctx.input(|input| input.key_down(key))
        }
        if pressed(ctx, egui::Key::W) {
            dpos += vec3(0., 0., 1.);
        }
        if pressed(ctx, egui::Key::S) {
            dpos += vec3(0., 0., -1.);
        }
        if pressed(ctx, egui::Key::A) {
            dpos += vec3(-1., 0., 0.);
        }
        if pressed(ctx, egui::Key::D) {
            dpos += vec3(1., 0., 0.);
        }
        if pressed(ctx, egui::Key::Space) {
            dpos += vec3(0., 1., 0.);
        }
        if pressed(ctx, egui::Key::Tab) {
            dpos += vec3(0., -1., 0.);
        }

        if pressed(ctx, egui::Key::I) {
            dpitch += rot_speed;
        }
        if pressed(ctx, egui::Key::K) {
            dpitch -= rot_speed;
        }
        if pressed(ctx, egui::Key::J) {
            dyaw += rot_speed;
        }
        if pressed(ctx, egui::Key::L) {
            dyaw -= rot_speed;
        }

        if dpos != vec3(0., 0., 0.) || dpitch != cgmath::Rad::zero() || dyaw != cgmath::Rad::zero()
        {
            self.world.camera.pitch += dpitch * dt;
            self.world.camera.yaw += dyaw * dt;
            self.world.camera.pos += (self
                .world
                .camera
                .yaw_mat()
                .invert()
                .expect("Bad View Matrix")
                * dpos)
                * dt
                * move_speed;
            ctx.request_repaint()
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.view.rect = ui.available_rect_before_wrap();
            self.draw_world(ui);
        });
    }
}
