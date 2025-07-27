#include "mgr.hpp"
#include "types.hpp"
#include "consts.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace nb = nanobind;

namespace madEscape {

// This file creates the python bindings used by the learning code.
// Refer to the nanobind documentation for more details on these functions.
NB_MODULE(madrona_escape_room, m) {
    // Each simulator has a madrona submodule that includes base types
    // like madrona::py::Tensor and madrona::py::PyExecMode.
    madrona::py::setupMadronaSubmodule(m);

    // Export observation sizes as module constants
    m.attr("SELF_OBSERVATION_SIZE") = nb::int_(madEscape::SelfObservationFloatCount);
    m.attr("STEPS_REMAINING_SIZE") = nb::int_(madEscape::StepsRemainingCount);
    m.attr("AGENT_ID_SIZE") = nb::int_(madEscape::AgentIDDimension);
    m.attr("TOTAL_OBSERVATION_SIZE") = nb::int_(madEscape::TotalObservationSize);
    
    // Export simulation constants
    m.attr("NUM_AGENTS") = nb::int_(madEscape::consts::numAgents);

    // Create action constants submodule
    auto action_module = m.def_submodule("action");
    
    // Export move_amount constants
    auto move_amount_module = action_module.def_submodule("move_amount");
    move_amount_module.attr("STOP") = nb::int_(madEscape::consts::action::move_amount::STOP);
    move_amount_module.attr("SLOW") = nb::int_(madEscape::consts::action::move_amount::SLOW);
    move_amount_module.attr("MEDIUM") = nb::int_(madEscape::consts::action::move_amount::MEDIUM);
    move_amount_module.attr("FAST") = nb::int_(madEscape::consts::action::move_amount::FAST);
    
    // Export move_angle constants
    auto move_angle_module = action_module.def_submodule("move_angle");
    move_angle_module.attr("FORWARD") = nb::int_(madEscape::consts::action::move_angle::FORWARD);
    move_angle_module.attr("FORWARD_RIGHT") = nb::int_(madEscape::consts::action::move_angle::FORWARD_RIGHT);
    move_angle_module.attr("RIGHT") = nb::int_(madEscape::consts::action::move_angle::RIGHT);
    move_angle_module.attr("BACKWARD_RIGHT") = nb::int_(madEscape::consts::action::move_angle::BACKWARD_RIGHT);
    move_angle_module.attr("BACKWARD") = nb::int_(madEscape::consts::action::move_angle::BACKWARD);
    move_angle_module.attr("BACKWARD_LEFT") = nb::int_(madEscape::consts::action::move_angle::BACKWARD_LEFT);
    move_angle_module.attr("LEFT") = nb::int_(madEscape::consts::action::move_angle::LEFT);
    move_angle_module.attr("FORWARD_LEFT") = nb::int_(madEscape::consts::action::move_angle::FORWARD_LEFT);
    
    // Export rotate constants
    auto rotate_module = action_module.def_submodule("rotate");
    rotate_module.attr("FAST_LEFT") = nb::int_(madEscape::consts::action::rotate::FAST_LEFT);
    rotate_module.attr("SLOW_LEFT") = nb::int_(madEscape::consts::action::rotate::SLOW_LEFT);
    rotate_module.attr("NONE") = nb::int_(madEscape::consts::action::rotate::NONE);
    rotate_module.attr("SLOW_RIGHT") = nb::int_(madEscape::consts::action::rotate::SLOW_RIGHT);
    rotate_module.attr("FAST_RIGHT") = nb::int_(madEscape::consts::action::rotate::FAST_RIGHT);

    nb::class_<Manager> (m, "SimManager")
        .def("__init__", [](Manager *self,
                            madrona::py::PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t rand_seed,
                            bool auto_reset,
                            bool enable_batch_renderer) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .randSeed = (uint32_t)rand_seed,
                .autoReset = auto_reset,
                .enableBatchRenderer = enable_batch_renderer,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("rand_seed"),
           nb::arg("auto_reset"),
           nb::arg("enable_batch_renderer") = false)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("self_observation_tensor", &Manager::selfObservationTensor)
        .def("steps_remaining_tensor", &Manager::stepsRemainingTensor)
        .def("progress_tensor", &Manager::progressTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
    ;
}

}
