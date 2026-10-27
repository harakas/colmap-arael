# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "estimators")

add_subdirectory(cost_functions)
add_subdirectory(solvers)

set(ESTIMATORS_SRCS
    alignment.h alignment.cc
    bundle_adjustment.h bundle_adjustment.cc
    bundle_adjustment_caspar.h bundle_adjustment_caspar.cc
    bundle_adjustment_ceres.h bundle_adjustment_ceres.cc
    ceres_loss_function.h ceres_loss_function.cc
    coordinate_frame.h coordinate_frame.cc
    covariance.h covariance.cc
    fundamental_matrix_degensac.h fundamental_matrix_degensac.cc
    generalized_pose.h generalized_pose.cc
    global_positioning.h global_positioning.cc
    gravity_refinement.h gravity_refinement.cc
    pose.h pose.cc
    rotation_averaging.h rotation_averaging.cc
    rotation_averaging_ceres.h rotation_averaging_ceres.cc
    rotation_averaging_l1_irls.h rotation_averaging_l1_irls.cc
    triangulation.h triangulation.cc
    two_view_geometry.h two_view_geometry.cc
    view_graph_calibration.h view_graph_calibration.cc
)

if(ARAEL_ENABLED)
    list(APPEND ESTIMATORS_SRCS
        bundle_adjustment_arael.h
        bundle_adjustment_arael.cc
    )
endif()

COLMAP_ADD_LIBRARY(
    NAME colmap_estimators
    SRCS ${ESTIMATORS_SRCS}
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_math
        colmap_feature_types
        colmap_geometry
        colmap_sensor
        colmap_image
        colmap_scene
        colmap_optim
        colmap_estimators_cost_functions
        colmap_estimators_solvers
        Eigen3::Eigen
        Ceres::ceres
        PoseLib::PoseLib
)

if(CASPAR_ENABLED)
    target_link_libraries(colmap_estimators PUBLIC caspar_lib_core)
endif()

if(ARAEL_ENABLED)
    # The arael model crate: cargo builds its C library, the generated
    # C++ headers wrap it. Needs a Rust toolchain.
    add_subdirectory(arael/cxx)
    # The cargo-built library is an imported target, which cannot be
    # exported. Wrap it in an interface library that links the imported
    # target in the build tree and the installed archive after install.
    get_target_property(_arael_capi_lib arael_colmap_arael IMPORTED_LOCATION)
    get_filename_component(_arael_capi_lib_name ${_arael_capi_lib} NAME)
    add_library(colmap_arael INTERFACE)
    target_link_libraries(colmap_arael INTERFACE
        $<BUILD_INTERFACE:arael::colmap_arael>
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}/${_arael_capi_lib_name}>)
    if(UNIX)
        target_link_libraries(colmap_arael INTERFACE
            $<INSTALL_INTERFACE:pthread>
            $<INSTALL_INTERFACE:dl>
            $<INSTALL_INTERFACE:m>)
    endif()
    install(FILES ${_arael_capi_lib} DESTINATION ${CMAKE_INSTALL_LIBDIR})
    target_link_libraries(colmap_estimators PUBLIC colmap_arael)
endif()

if(CUDA_ENABLED)
    target_link_libraries(colmap_estimators PUBLIC colmap_util_cuda)
endif()

COLMAP_ADD_TEST(
    NAME alignment_test
    SRCS alignment_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME bundle_adjustment_test
    SRCS bundle_adjustment_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME bundle_adjustment_ceres_test
    SRCS bundle_adjustment_ceres_test.cc
    LINK_LIBS colmap_estimators
)
if(CASPAR_ENABLED)
    COLMAP_ADD_TEST(
        NAME bundle_adjustment_caspar_test
        SRCS bundle_adjustment_caspar_test.cc
        LINK_LIBS colmap_estimators
    )
endif()
if(ARAEL_ENABLED)
    COLMAP_ADD_TEST(
        NAME bundle_adjustment_arael_test
        SRCS bundle_adjustment_arael_test.cc
        LINK_LIBS colmap_estimators
    )
endif()
COLMAP_ADD_TEST(
    NAME coordinate_frame_test
    SRCS coordinate_frame_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME covariance_test
    SRCS covariance_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME fundamental_matrix_degensac_test
    SRCS fundamental_matrix_degensac_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME generalized_pose_test
    SRCS generalized_pose_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME pose_test
    SRCS pose_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME two_view_geometry_test
    SRCS two_view_geometry_test.cc
    LINK_LIBS colmap_estimators
)
COLMAP_ADD_TEST(
    NAME view_graph_calibration_test
    SRCS view_graph_calibration_test.cc
    LINK_LIBS colmap_estimators
)

COLMAP_ADD_TEST(
    NAME rotation_averaging_test
    SRCS rotation_averaging_test.cc
    LINK_LIBS colmap_estimators
)

COLMAP_ADD_TEST(
    NAME rotation_averaging_ceres_test
    SRCS rotation_averaging_ceres_test.cc
    LINK_LIBS colmap_estimators
)

COLMAP_ADD_TEST(
    NAME gravity_refinement_test
    SRCS gravity_refinement_test.cc
    LINK_LIBS colmap_estimators
)

COLMAP_ADD_TEST(
    NAME global_positioning_test
    SRCS global_positioning_test.cc
    LINK_LIBS colmap_estimators
)
