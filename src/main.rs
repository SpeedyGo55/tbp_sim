#![windows_subsystem = "windows"]

use lazy_static::lazy_static;
use nannou::prelude::*;
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use std::path::{PathBuf};

const TIME_STEP: f32 = 0.01;

const SIZE: f32 = 250.0;


lazy_static! {
    static ref BODIES: Vec<Body> = {
        let path = std::env::current_exe()
            .expect("Unable to get current executable path")
            .parent()
            .expect("Unable to get parent directory")
            .join("config.json");
        let mut bodies = load_bodies_json(Box::new(path));
        for body in &mut bodies {
            body.pos = body.pos * SIZE;
            body.vel = body.vel * SIZE;
        }
        bodies
    };
}

fn load_bodies_json(filepath: Box<PathBuf>) -> Vec<Body> {
    let file = std::fs::File::open(filepath.as_path()).expect("Unable to open file");
    let reader = std::io::BufReader::new(file);
    let bodies: Vec<Body> = serde_json::from_reader(reader).expect("Unable to parse JSON");
    bodies
}

fn save_bodies_json(filepath: Box<PathBuf>, bodies: &Vec<Body>) {
    if let Some(parent) = filepath.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("Failed to create directories: {}", e);
            return;
        }
    }
    let file = match std::fs::File::create(filepath.as_path()) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Unable to create file: {}", e);
            return;
        }
    };
    let writer = std::io::BufWriter::new(file);
    if let Err(e) = serde_json::to_writer(writer, bodies) {
        eprintln!("Unable to write JSON: {}", e);
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Body {
    pos: Vec2,
    vel: Vec2,
    acc: Vec2,
    mass: f32,
    color: Rgb<u8>,
}

impl Body {
    #[cfg_attr(not(test), allow(dead_code))]
    fn new(pos: Vec2, vel: Vec2, mass: f32, color: Rgb<u8>) -> Self {
        Self {
            pos,
            vel,
            acc: Vec2::ZERO,
            mass,
            color,
        }
    }

    fn draw(&self, draw: &Draw) {
        let r = (self.mass / f32::PI()).sqrt();
        let r = r * 20.0;
        draw.ellipse()
            .x_y(self.pos.x, self.pos.y)
            .w_h(r * 2.0 , r * 2.0)
            .rgb(
                self.color.red as f32 / 255.0,
                self.color.green as f32 / 255.0,
                self.color.blue as f32 / 255.0,
            );
    }
}

/// Newtonian pairwise gravity with Plummer softening: a = G * sum_j m_j * r_ij / (|r_ij|^2 + eps^2)^(3/2).
/// Softening keeps the acceleration finite and smooth as r -> 0, instead of
/// the discontinuous slope produced by clamping r^2 to a floor.
fn compute_accelerations(bodies: &[Body], g: f32, softening: f32) -> Vec<Vec2> {
    let mut accs = vec![Vec2::ZERO; bodies.len()];
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let dir = bodies[j].pos - bodies[i].pos;
            let dist_sq = dir.length_squared() + softening;
            let inv_dist3 = dist_sq.powf(-1.5);
            // a_i += G * m_j * dir / |dir|^3 ; a_j -= G * m_i * dir / |dir|^3 (Newton's 3rd law)
            accs[i] += dir * (g * bodies[j].mass * inv_dist3);
            accs[j] -= dir * (g * bodies[i].mass * inv_dist3);
        }
    }
    accs
}

/// One velocity-Verlet step: symplectic and time-reversible, so orbital
/// energy oscillates around its true value instead of drifting away from it
/// the way explicit/semi-implicit Euler does over long integrations.
fn step_verlet(bodies: &mut [Body], g: f32, softening: f32, dt: f32) {
    for body in bodies.iter_mut() {
        body.pos += body.vel * dt + body.acc * (0.5 * dt * dt);
    }
    let new_accs = compute_accelerations(bodies, g, softening);
    for (body, &new_acc) in bodies.iter_mut().zip(new_accs.iter()) {
        body.vel += (body.acc + new_acc) * (0.5 * dt);
        body.acc = new_acc;
    }
}

/// Advances by exactly `total_dt` using as many velocity-Verlet substeps as
/// the current dynamics need, instead of one fixed-size step.
///
/// A fixed step that is fine for widely separated bodies becomes wildly
/// inaccurate the moment two bodies pass close together: the acceleration
/// spikes, a single large step overshoots the encounter, and the resulting
/// velocity kick injects energy that isn't there in the real system (a
/// close flyby gets misintegrated into an unphysical high-speed ejection).
/// Standard N-body codes (e.g. Aarseth-style integrators) address this with
/// an acceleration-based timestep criterion: dt ~ eta * sqrt(softening_length / |a|).
/// Shrinking dt only while some body's acceleration is large keeps ordinary
/// frames cheap (one substep) while resolving close encounters correctly.
fn step_adaptive(bodies: &mut [Body], g: f32, softening: f32, total_dt: f32) {
    const ETA: f32 = 0.01;
    const MAX_SUBSTEPS: u32 = 50_000;
    let softening_length = softening.sqrt();
    let mut remaining = total_dt;
    let mut substeps = 0;
    while remaining > 1e-6 && substeps < MAX_SUBSTEPS {
        let a_max = bodies.iter().map(|b| b.acc.length()).fold(0.0f32, f32::max);
        let dt_stable = if a_max > f32::EPSILON {
            ETA * (softening_length / a_max).sqrt()
        } else {
            remaining
        };
        let dt = dt_stable.min(remaining);
        step_verlet(bodies, g, softening, dt);
        remaining -= dt;
        substeps += 1;
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn total_momentum(bodies: &[Body]) -> Vec2 {
    bodies.iter().fold(Vec2::ZERO, |acc, b| acc + b.vel * b.mass)
}

#[cfg_attr(not(test), allow(dead_code))]
fn total_energy(bodies: &[Body], g: f32) -> f32 {
    let kinetic: f32 = bodies
        .iter()
        .map(|b| 0.5 * b.mass * b.vel.length_squared())
        .sum();
    let mut potential = 0.0f32;
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let dist = (bodies[j].pos - bodies[i].pos).length();
            potential -= g * bodies[i].mass * bodies[j].mass / dist;
        }
    }
    kinetic + potential
}

struct Model {
    initial_bodies: Vec<Body>,
    bodies: Vec<Body>,
    running: bool,
    space_down: bool,
    selected_body: Option<usize>,
}

fn model(_app: &App) -> Model {
    let mut bodies = BODIES.clone();
    seed_accelerations(&mut bodies, G_SCALED, SOFTENING_SCALED);
    Model {
        initial_bodies: bodies.clone(),
        bodies,
        running: true,
        space_down: false,
        selected_body: None,
    }
}

/// Velocity Verlet needs the acceleration at the *start* of each step before
/// it can advance positions, so bodies loaded from disk (whose `acc` field is
/// stale or zero) must have it computed once up front.
fn seed_accelerations(bodies: &mut [Body], g: f32, softening: f32) {
    let accs = compute_accelerations(bodies, g, softening);
    for (body, acc) in bodies.iter_mut().zip(accs) {
        body.acc = acc;
    }
}

const G_SCALED: f32 = SIZE * SIZE * SIZE;
// SOFTENING (5.0 in squared-position units) is calibrated for the app's
// on-screen coordinates, where SIZE=250 makes typical separations squared
// on the order of 1e4-1e5 - negligible next to it except during a genuine
// close encounter. Tests that use small, unscaled coordinates must pass
// their own much smaller (or zero) softening instead of this constant.
const SOFTENING_SCALED: f32 = 5.0;

fn update(app: &App, model: &mut Model, _update: Update) {
    if app.keys.down.contains(&Key::Space) && !model.space_down {
        model.running = !model.running;
        model.space_down = true;
    } else if !app.keys.down.contains(&Key::Space) {
        model.space_down = false;
    }

    if app.keys.down.contains(&Key::R) {
        model.bodies = model.initial_bodies.clone();
    }

    if app.mouse.buttons.left().is_down() {
        match model.selected_body {
            Some(index) => {
                let body = &mut model.bodies[index];
                body.pos = app.mouse.position();
                body.vel = Vec2::ZERO;
                // The body just jumped to a new position, so its stored
                // acceleration (from the old position) is stale; without a
                // reseed the next Verlet step blends a wrong "old" half-kick
                // in with the correct new one.
                seed_accelerations(&mut model.bodies, G_SCALED, SOFTENING_SCALED);
            }
            None => {
                for (i, body) in model.bodies.iter_mut().enumerate() {
                    let r = (body.mass / f32::PI()).sqrt() * 20.0;
                    if body.pos.distance(app.mouse.position()) < r + 1.0 {
                        model.selected_body = Some(i);
                        break;
                    }
                }
            }
        }
    } else {
        model.selected_body = None;
    }

    if app.keys.down.contains(&Key::S) {
        if let Some(path) = FileDialog::new()
            .set_title("Save Body")
            .add_filter("JSON", &["json"])
            .save_file()
        {
            if !path.exists() {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                        eprintln!("Failed to create directories: {}", e);
                    });
                }
            }
            let mut bodies = model.bodies.clone();
            for body in &mut bodies {
                body.pos = body.pos / SIZE;
                body.vel = body.vel / SIZE;
            }

            save_bodies_json(Box::new(path), &bodies);
        }
    }

    if app.keys.down.contains(&Key::L) {
        if let Some(path) = FileDialog::new()
            .set_title("Load Body")
            .add_filter("JSON", &["json"])
            .pick_file()
        {
            let mut bodies = load_bodies_json(Box::new(path));
            for body in &mut bodies {
                body.pos = body.pos * SIZE;
                body.vel = body.vel * SIZE;
            }
            seed_accelerations(&mut bodies, G_SCALED, SOFTENING_SCALED);
            model.bodies = bodies.clone();
            model.initial_bodies = bodies.clone();
        }
    }

    if model.running {
        step_adaptive(&mut model.bodies, G_SCALED, SOFTENING_SCALED, TIME_STEP);
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    let win = app.window_rect();
    if frame.nth() == 0 {
        draw.background().color(BLACK);
    } else {
        draw.rect()
            .w_h(win.w(), win.h())
            .color(srgba(0.0, 0.0, 0.0, 0.05));
    }

    for body in &model.bodies {
        body.draw(&draw);
    }

    draw.to_frame(app, &frame).unwrap();
}

fn main() {
    nannou::app(model)
        .update(update)
        .simple_window(view)
        .fullscreen()
        .run();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn body(pos: Vec2, vel: Vec2, mass: f32) -> Body {
        Body::new(pos, vel, mass, Rgb::new(0, 0, 0))
    }

    fn run(bodies: &mut [Body], g: f32, dt: f32, steps: u32) {
        for _ in 0..steps {
            step_verlet(bodies, g, 0.0, dt);
        }
    }

    /// Gravity between any pair must be equal and opposite (Newton's third
    /// law), independent of the two masses.
    #[test]
    fn newtons_third_law_holds_per_pair() {
        let bodies = vec![
            body(vec2(0.0, 0.0), Vec2::ZERO, 3.0),
            body(vec2(4.0, 0.0), Vec2::ZERO, 7.0),
        ];
        let accs = compute_accelerations(&bodies, 1.0, 0.0);
        let force_on_0 = accs[0] * bodies[0].mass;
        let force_on_1 = accs[1] * bodies[1].mass;
        assert!((force_on_0 + force_on_1).length() < 1e-4);
    }

    /// A closed N-body system has no external forces, so total momentum
    /// (sum of m*v) must stay constant for all time, not just at t=0.
    #[test]
    fn total_momentum_is_conserved() {
        let mut bodies = vec![
            body(vec2(-1.0, 0.0), vec2(0.0, 0.3), 1.0),
            body(vec2(1.0, 0.3), vec2(0.1, -0.2), 2.0),
            body(vec2(0.5, -1.2), vec2(-0.2, 0.1), 1.5),
            body(vec2(-0.3, 0.8), vec2(0.05, 0.05), 0.5),
        ];
        seed_accelerations(&mut bodies, 1.0, 0.0);
        let p0 = total_momentum(&bodies);
        run(&mut bodies, 1.0, 0.001, 2000);
        let p1 = total_momentum(&bodies);
        assert!((p1 - p0).length() < 1e-3, "momentum drifted: {p0:?} -> {p1:?}");
    }

    /// Velocity Verlet is symplectic: energy should oscillate in a bounded
    /// band around the true value rather than drifting away, even over many
    /// steps of a genuinely chaotic-adjacent orbit.
    #[test]
    fn total_energy_stays_bounded_over_long_integration() {
        let mut bodies = vec![
            body(vec2(1.0, 0.0), vec2(0.0, 1.0), 1.0),
            body(vec2(-1.0, 0.0), vec2(0.0, -1.0), 1.0),
        ];
        seed_accelerations(&mut bodies, 1.0, 0.0);
        let e0 = total_energy(&bodies, 1.0);
        let mut max_dev: f32 = 0.0;
        for _ in 0..20_000 {
            step_verlet(&mut bodies, 1.0, 0.0, 0.001);
            let e = total_energy(&bodies, 1.0);
            max_dev = max_dev.max((e - e0).abs() / e0.abs());
        }
        assert!(max_dev < 0.01, "energy drifted by {:.4}% of |E0|", max_dev * 100.0);
    }

    /// Two-body circular orbit: at the analytic Kepler period the orbiting
    /// body must be back near its starting position, and Kepler's law
    /// T = 2*pi*sqrt(r^3/(G*M)) must hold for the chosen radius/speed.
    #[test]
    fn two_body_circular_orbit_matches_kepler_period() {
        let g = 1.0;
        let m_central = 1000.0;
        let r = 10.0;
        let v_circular = (g * m_central / r).sqrt();
        let mut bodies = vec![
            body(vec2(0.0, 0.0), Vec2::ZERO, m_central),
            body(vec2(r, 0.0), vec2(0.0, v_circular), 1e-6),
        ];
        seed_accelerations(&mut bodies, g, 0.0);

        let period = 2.0 * PI * (r.powi(3) / (g * m_central)).sqrt();
        let dt = period / 20_000.0;
        let steps = 20_000u32;
        run(&mut bodies, g, dt, steps);

        let final_pos = bodies[1].pos;
        let final_radius = final_pos.length();
        assert!(
            (final_radius - r).abs() / r < 0.01,
            "orbit radius drifted: expected {r}, got {final_radius}"
        );
        let angle_error = final_pos.angle().rem_euclid(2.0 * PI);
        let wrap_error = angle_error.min(2.0 * PI - angle_error);
        assert!(
            wrap_error < 0.05,
            "did not return to start after one Kepler period, angle error = {wrap_error} rad"
        );
    }

    /// N equal masses on a regular polygon are a valid central configuration
    /// only at a specific angular velocity (Moulton's formula); at that
    /// speed the ring rotates rigidly and its radius stays constant.
    #[test]
    fn ring_configuration_holds_shape_at_correct_angular_velocity() {
        let n = 10;
        let radius = 1.0;
        let s: f32 = (1..n).map(|k| 1.0 / ((k as f32 * PI / n as f32).sin())).sum();
        let omega = (s / 4.0).sqrt();

        let mut bodies: Vec<Body> = (0..n)
            .map(|k| {
                let theta = 2.0 * PI * k as f32 / n as f32;
                let pos = vec2(radius * theta.cos(), radius * theta.sin());
                let vel = vec2(-omega * pos.y, omega * pos.x);
                body(pos, vel, 1.0)
            })
            .collect();
        seed_accelerations(&mut bodies, 1.0, 0.0);

        run(&mut bodies, 1.0, 0.0005, 4000);

        for b in &bodies {
            assert!(
                (b.pos.length() - radius).abs() / radius < 0.02,
                "ring body drifted off its orbit radius: {}",
                b.pos.length()
            );
        }
    }

    /// Regression: the shipped many.json ring used unit angular velocity,
    /// far below the ~1.965 rad/s Moulton's-formula requires for a 10-body
    /// ring of radius 1 (mass 1, G=1) — that config collapses instead of
    /// holding its shape. This pins the correct value so a future edit to
    /// many.json can be checked against it.
    #[test]
    fn moulton_omega_for_decagon_is_not_one() {
        let n = 10;
        let s: f32 = (1..n).map(|k| 1.0 / ((k as f32 * PI / n as f32).sin())).sum();
        let omega = (s / 4.0).sqrt();
        assert!((omega - 1.9653116).abs() < 1e-4);
    }

    /// Regression for the shipped config.json scenario: a light body starts
    /// close enough to a mass-100 body that it swings past at high
    /// acceleration within the first few simulated seconds. A single fixed
    /// step of TIME_STEP there overshoots the encounter and injects energy
    /// by orders of magnitude (the app's original bug); step_adaptive must
    /// keep total energy essentially unchanged across the same encounter.
    #[test]
    fn adaptive_stepping_survives_close_encounter_that_breaks_fixed_step() {
        let g = SIZE * SIZE * SIZE;
        let softening = 5.0f32;
        let make_bodies = || {
            vec![
                body(vec2(0.97000436, -0.24308753) * SIZE, vec2(0.46620368, 0.43236573) * SIZE, 1.0),
                body(vec2(-0.97000436, 0.24308753) * SIZE, vec2(0.46620368, 0.43236573) * SIZE, 1.0),
                body(vec2(1.0, 0.0) * SIZE, vec2(-0.93240737, -0.86473146) * SIZE, 1.0),
                body(vec2(0.0, 0.0) * SIZE, Vec2::ZERO, 100.0),
            ]
        };

        // Fixed-step baseline reproduces the original blow-up.
        let mut fixed_bodies = make_bodies();
        seed_accelerations(&mut fixed_bodies, g, softening);
        let e0 = total_energy(&fixed_bodies, g);
        for _ in 0..500 {
            step_verlet(&mut fixed_bodies, g, softening, TIME_STEP);
        }
        let fixed_drift = (total_energy(&fixed_bodies, g) - e0).abs() / e0.abs();
        assert!(fixed_drift > 1.0, "expected the fixed-step baseline to blow up, drift={fixed_drift}");

        // Adaptive stepping over the same total simulated time must not.
        let mut adaptive_bodies = make_bodies();
        seed_accelerations(&mut adaptive_bodies, g, softening);
        let e0 = total_energy(&adaptive_bodies, g);
        for _ in 0..500 {
            step_adaptive(&mut adaptive_bodies, g, softening, TIME_STEP);
        }
        let adaptive_drift = (total_energy(&adaptive_bodies, g) - e0).abs() / e0.abs();
        assert!(
            adaptive_drift < 0.01,
            "adaptive stepping should conserve energy through the encounter, drift={adaptive_drift}"
        );
    }
}
