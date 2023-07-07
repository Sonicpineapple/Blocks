use cgmath::num_traits::ToPrimitive;
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
        Self {
            world: World {
                blocks: vec![
                    Wireframe::block(vec3(-1., -1., 1.), vec3(1., 1., 1.), egui::Color32::GREEN),
                    Wireframe::block(vec3(-2., 1., 1.), vec3(2., 2., 1.), egui::Color32::RED),
                    Wireframe::block(vec3(1., 1., 2.), vec3(1., 2., 3.), egui::Color32::BLUE),
                ],
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

                let mut a = to_screen(a);
                let mut b = to_screen(b);

                let xmin = self.view.rect.left();
                let xmax = self.view.rect.right();
                let ymin = self.view.rect.top();
                let ymax = self.view.rect.bottom();

                if b.x < a.x {
                    std::mem::swap(&mut a, &mut b);
                }
                if b.x < xmin {
                    continue;
                }
                if a.x < xmin {
                    let da = a.x - xmin;
                    let db = b.x - xmin;
                    a = ((db * a.to_vec2() - da * b.to_vec2()) / (db - da)).to_pos2()
                }
                if b.x > a.x {
                    std::mem::swap(&mut a, &mut b);
                }
                if b.x > xmax {
                    continue;
                }
                if a.x > xmax {
                    let da = a.x - xmax;
                    let db = b.x - xmax;
                    a = ((db * a.to_vec2() - da * b.to_vec2()) / (db - da)).to_pos2()
                }
                if b.y < a.y {
                    std::mem::swap(&mut a, &mut b);
                }
                if b.y < ymin {
                    continue;
                }
                if a.y < ymin {
                    let da = a.y - ymin;
                    let db = b.y - ymin;
                    a = ((db * a.to_vec2() - da * b.to_vec2()) / (db - da)).to_pos2()
                }
                if b.y > a.y {
                    std::mem::swap(&mut a, &mut b);
                }
                if b.y > ymax {
                    continue;
                }
                if a.y > ymax {
                    let da = a.y - ymax;
                    let db = b.y - ymax;
                    a = ((db * a.to_vec2() - da * b.to_vec2()) / (db - da)).to_pos2()
                }

                ui.painter().line_segment([a, b], (2.0, block.col))
            }
        }
    }
}

#[derive(Debug, Clone)]
struct World {
    blocks: Vec<Wireframe>,
    camera: Camera,
}
impl World {
    fn add_block(&mut self, block: Wireframe) {
        self.blocks.push(block);
    }
}

#[derive(Debug, Clone)]
struct Wireframe {
    verts: Vec<Vector3<f32>>,
    edges: Vec<[usize; 2]>,
    col: egui::Color32,
}
impl Wireframe {
    fn block(min: Vector3<f32>, size: Vector3<f32>, col: egui::Color32) -> Self {
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
        Self { verts, edges, col }
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
    fn new(full_rect: egui::Rect) -> Viewport {
        let cen = full_rect.center();
        let size = full_rect.size().min_elem() * 0.9;
        Self {
            rect: egui::Rect::from_center_size(cen, egui::vec2(1., 1.) * size),
        }
    }
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
        if pressed(ctx, egui::Key::E) {
            dpos += vec3(0., 0., 1.);
        }
        if pressed(ctx, egui::Key::D) {
            dpos += vec3(0., 0., -1.);
        }
        if pressed(ctx, egui::Key::S) {
            dpos += vec3(-1., 0., 0.);
        }
        if pressed(ctx, egui::Key::F) {
            dpos += vec3(1., 0., 0.);
        }
        if pressed(ctx, egui::Key::Space) {
            dpos += vec3(0., 1., 0.);
        }
        if pressed(ctx, egui::Key::A) {
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

        egui::SidePanel::left("shape_panel").show(ctx, |ui| {
            let paths = std::fs::read_dir(
                "C:/Users/scare/OneDrive/Documents/Cubes and Puzzles/blocks/data/lib/reg3/",
            )
            .unwrap();
            let _labels = paths
                .filter_map(Result::ok)
                .map(|path| {
                    (
                        egui::Label::new(path.file_name().to_str().unwrap())
                            .sense(egui::Sense::click()),
                        path,
                    )
                })
                .for_each(|(label, path)| {
                    if ui.add(label).clicked() {
                        #[derive(Debug, Clone, PartialEq)]
                        enum Value {
                            Num(f32),
                            List(Vec<Value>),
                            Open,
                            Edge((usize, usize)),
                            Face(Vec<usize>),
                        }
                        impl Value {
                            fn float(&self) -> f32 {
                                if let Value::Num(v) = self {
                                    *v
                                } else {
                                    panic!("Expected numeric value but got {self:?}")
                                }
                            }
                        }

                        let str = std::fs::read_to_string(path.path()).expect("Bad file");
                        let rgx = regex::Regex::new(r"[\d\.-]+|\[|\]|[^\[\]\s]+").unwrap();
                        let def = rgx.find_iter(&str);
                        let mut stack: Vec<Value> = vec![];
                        for token in def {
                            match token.as_str() {
                                "[" => stack.push(Value::Open),
                                "]" => {
                                    let i = stack
                                        .iter()
                                        .rposition(|v| *v == Value::Open)
                                        .expect("Extra ] in file");
                                    let v = Value::List(stack[(i + 1)..].to_vec());
                                    stack = stack[..i].to_vec();
                                    stack.push(v);
                                }
                                "edge" => {
                                    if let (Value::Num(b), Value::Num(a)) =
                                        (stack.pop().unwrap(), stack.pop().unwrap())
                                    {
                                        stack.push(Value::Edge((
                                            a.to_usize().expect("Bad index when making edge"),
                                            b.to_usize().expect("Bad index when making edge"),
                                        )))
                                    }
                                }
                                "face" => {
                                    if let (Value::List(norm), Value::List(inds)) =
                                        (stack.pop().unwrap(), stack.pop().unwrap())
                                    {
                                    }
                                }
                                "shape" => break,
                                _ => {
                                    if let Ok(x) = token.as_str().parse() {
                                        stack.push(Value::Num(x))
                                    }
                                }
                            }
                        }
                        let verts = if let Value::List(v) = &stack[0] {
                            v.clone()
                        } else {
                            panic!()
                        };
                        let edges = if let Value::List(v) = &stack[1] {
                            v.clone()
                        } else {
                            panic!()
                        };

                        let verts = verts
                            .iter()
                            .map(|v| {
                                if let Value::List(v) = v {
                                    vec3(v[0].float(), v[1].float(), v[2].float())
                                } else {
                                    panic!()
                                }
                            })
                            .collect_vec();
                        let edges = edges
                            .iter()
                            .map(|v| {
                                if let Value::Edge((i, j)) = v {
                                    [*i, *j]
                                } else {
                                    panic!()
                                }
                            })
                            .collect_vec();

                        let shape = Wireframe {
                            verts,
                            edges,
                            col: egui::Color32::GOLD,
                        };
                        self.world.add_block(shape);

                        // magic the path in
                    }
                });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.view = Viewport::new(ui.available_rect_before_wrap());
            self.draw_world(ui);
            ui.painter()
                .rect_stroke(self.view.rect, 0., (2., egui::Color32::GRAY));
        });
    }
}
