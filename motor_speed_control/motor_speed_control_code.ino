#include <Servo.h>

Servo esc;  // ESC bir servo gibi kontrol edilir

void setup() {
  Serial.begin(115200);
  esc.attach(9);  // ESC D9 pinine bağlı

  esc.writeMicroseconds(1000);  // ESC'yi armla
  Serial.println("ESC hazır. 0–100 arası yüzde değeri girin.");
  delay(3000);
}

void loop() {
  if (Serial.available()) {
    int percent = Serial.parseInt();

    if (percent >= 0 && percent <= 100) {
      int pwm = map(percent, 0, 100, 1000, 2000);  // 1000–2000 µs
      esc.writeMicroseconds(pwm);

      Serial.print("PWM %: ");
      Serial.print(percent);
      Serial.print(" → PWM µs: ");
      Serial.println(pwm);
    } else {
      Serial.println("Hatalı giriş! 0–100 arası sayı girin.");
    }
  }
}
