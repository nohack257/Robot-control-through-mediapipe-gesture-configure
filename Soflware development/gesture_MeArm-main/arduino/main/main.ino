#include <Servo.h>

Servo servo[4];
const int default_angle[4] = {75, 90, 90, 60};
const int MIN_ANGLE = 0;
const int MAX_ANGLE = 180;

void setup()
{
    Serial.begin(115200);
    // Using constants for pins improves readability
    const int servoPins[4] = {5, 6, 8, 7};
    
    for (size_t i = 0; i < 4; i++)
    {
        servo[i].attach(servoPins[i]);
        servo[i].write(default_angle[i]);
    }
}

byte angle[4];
byte pre_angle[4];
unsigned long lastCommandTime = millis();

void loop()
{
    if (Serial.available() >= 4)  // Only read if we have all 4 bytes
    {
        Serial.readBytes(angle, 4);
        
        // Validate and update servo positions
        for (size_t i = 0; i < 4; i++)
        {
            if (angle[i] != pre_angle[i] && 
                angle[i] >= MIN_ANGLE && 
                angle[i] <= MAX_ANGLE)
            {
                servo[i].write(angle[i]);
                pre_angle[i] = angle[i];
            }
        }
        lastCommandTime = millis();
    }

    // Return to default positions if no commands received for 1 second
    if (millis() - lastCommandTime > 1000)
    {
        for (size_t i = 0; i < 4; i++)
        {
            if (pre_angle[i] != default_angle[i])
            {
                servo[i].write(default_angle[i]);
                pre_angle[i] = default_angle[i];
            }
        }
    }
}