#include "test_framework.hpp"
#include "neuroflow/scheduler.hpp"
#include <cmath>

using namespace neuroflow;

TEST(CosineScheduler, Construction) {
    CosineScheduler sched(0.001f, 1000);
    EXPECT_NEAR(sched.lr_max_, 0.001f, 1e-8f);
    EXPECT_NEAR(sched.lr_min_, 0.001f * 0.1f, 1e-8f);
    EXPECT_EQ(sched.total_steps_, 1000u);
    EXPECT_GT(sched.warmup_steps_, 0u);
}

TEST(CosineScheduler, StepZeroLr) {
    CosineScheduler sched(0.001f, 1000, 0.1f, 0.01f);
    float lr = sched.get_lr(0);
    EXPECT_GT(lr, 0.0f);
    EXPECT_LT(lr, sched.lr_max_);
}

TEST(CosineScheduler, WarmupPhaseLinearGrowth) {
    CosineScheduler sched(0.001f, 1000, 0.1f, 0.1f);
    EXPECT_EQ(sched.warmup_steps_, 100u);
    EXPECT_TRUE(sched.get_phase(0) == CosineScheduler::Phase::WARMUP);
    EXPECT_TRUE(sched.get_phase(50) == CosineScheduler::Phase::WARMUP);

    float lr_50 = sched.get_lr(50);
    float lr_100 = sched.get_lr(100);
    EXPECT_GT(lr_100, lr_50);
    EXPECT_NEAR(lr_100, 0.001f, 1e-5f);
}

TEST(CosineScheduler, CosinePhaseDecay) {
    CosineScheduler sched(0.001f, 1000, 0.1f, 0.1f);
    EXPECT_TRUE(sched.get_phase(200) == CosineScheduler::Phase::COSINE);

    float lr_100 = sched.get_lr(100);
    float lr_500 = sched.get_lr(500);
    float lr_900 = sched.get_lr(900);
    EXPECT_GT(lr_100, lr_500);
    EXPECT_GT(lr_500, lr_900);
}

TEST(CosineScheduler, FinalLrNearMin) {
    CosineScheduler sched(0.001f, 1000, 0.1f, 0.1f);
    float lr_final = sched.get_lr(999);
    float lr_min = 0.001f * 0.1f;
    EXPECT_NEAR(lr_final, lr_min, lr_min * 0.15f);
}

TEST(CosineScheduler, DonePhase) {
    CosineScheduler sched(0.001f, 100);
    EXPECT_TRUE(sched.get_phase(100) == CosineScheduler::Phase::DONE);
    EXPECT_TRUE(sched.get_phase(200) == CosineScheduler::Phase::DONE);
}

int main() { RUN_ALL_TESTS(); }
