# N Body Problem Simulation
This is a Rust implementation of the N Body Problem simulation, which simulates the gravitational interactions between celestial bodies in a 2D space. The simulation uses an iterative approach to update the positions and velocities of the bodies over time. 
### Disclaimer
This is only a model of the N Body Problem. The simulation does not take into account many factors that would be present in a real-world scenario and I do not claim that this is an accurate simulation of the N Body Problem. The simulation is intended for educational purposes only and should not be used for any real-world applications.

## Features

- Simulates the gravitational interactions between celestial bodies in a 2D space.
- Uses an iterative approach to update the positions and velocities of the bodies over time.
- Savable state of the simulation to a file.
- Loadable state of the simulation from a file.
- Pausable simulation.
- Draggable bodies.
- Virtually unlimited number of bodies.

## Physics notes

- Gravity is pairwise Newtonian (`F = G*m1*m2/r^2`), with Newton's third law
  enforced exactly (every pair contributes equal and opposite force).
- Positions/velocities are integrated with velocity Verlet, a symplectic,
  time-reversible integrator. Unlike the semi-implicit Euler step this
  replaced, its energy error stays bounded over long runs instead of
  drifting away monotonically.
- Close encounters are handled by Plummer softening (`r^2 + eps^2` in the
  force denominator) plus adaptive substepping: when any body's
  acceleration spikes (a close pass), the frame's time step is
  automatically subdivided so the encounter is resolved accurately instead
  of injecting energy from an oversized step.
- `cargo test` exercises the physics engine directly: Newton's third law,
  conservation of momentum and energy, a two-body orbit checked against its
  analytic Kepler period, and an N-body ring configuration checked against
  Moulton's central-configuration formula for the angular velocity a
  regular polygon of equal masses needs to hold its shape.