#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;

int main()
{
    const int W = 800, H = 400;
    Mat img(H, W, CV_8UC3);

    const int N = 200;
    const double u = 1.0;
    const double Q = 0.01;
    const double R = 0.25;

    double x_true = 0.0, x_est = 0.0, P = 1.0;

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> w(0.0, sqrt(Q));
    normal_distribution<> v(0.0, sqrt(R));

    for (int k = 0; k < N; ++k)
    {
        img.setTo(Scalar(255,255,255));

        // 真値
        x_true += u + w(gen);
        double z = x_true + v(gen);

        // カルマン
        double x_pred = x_est + u;
        double P_pred = P + Q;
        double K = P_pred / (P_pred + R);
        x_est = x_pred + K * (z - x_pred);
        P = (1.0 - K) * P_pred;

        int x_t = static_cast<int>(x_true * 5);
        int x_m = static_cast<int>(z * 5);
        int x_k = static_cast<int>(x_est * 5);

        circle(img, {x_t, H/2 - 50}, 5, {0,255,0}, -1); // 真値
        circle(img, {x_m, H/2},      5, {0,0,255}, -1); // 観測
        circle(img, {x_k, H/2 + 50},  5, {255,0,0}, -1); // 推定

        imshow("Kalman Filter", img);
        if (waitKey(50) == 27) break;
    }
    return 0;
}
