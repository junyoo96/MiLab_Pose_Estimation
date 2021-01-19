package(default_visibility = ["//visibility:public"])

cc_library(
    name = "landmark_writer_calculator",
    srcs = ["landmark_writer_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:landmark_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "detection_info_write_plot_calculator",
    srcs = ["detection_info_write_plot_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:detection_cc_proto",
    ],
    alwayslink = 1,
)
