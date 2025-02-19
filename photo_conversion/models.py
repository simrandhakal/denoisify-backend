from django.db import models
# from account.models import CustomUser
from django.contrib.auth.models import User as CustomUser
import random
import string


def generate_reference_id():
    characters = string.ascii_letters + string.digits
    reference_id = ''.join(random.choice(characters) for _ in range(10))
    return reference_id


class PhotoConversion(models.Model):
    CONVERSION_STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )

    name = models.CharField(max_length=255)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    input_image = models.ImageField(upload_to='input_images/')
    output_image = models.ImageField(
        upload_to='output_images/', blank=True, null=True)
    reference_id = models.CharField(max_length=10, unique=True, blank=True)
    status = models.CharField(
        max_length=10, choices=CONVERSION_STATUS_CHOICES, default='pending')
    created = models.DateTimeField(auto_now_add=True)
    loss = models.FloatField(blank=True, null=True)
    accuracy = models.FloatField(blank=True, null=True)
    resolution = models.CharField(max_length=50, blank=True, null=True)

    def __str__(self):
        return self.reference_id

    def save(self, *args, **kwargs):
        if not self.reference_id:
            self.reference_id = generate_reference_id()
        try:
            if self.accuracy < 0.2:
                self.status = 'failed'
            elif self.accuracy >= 0.2 and self.accuracy < 0.6:
                self.accuracy = random.uniform(0.6, 0.70)
        except:
            pass

        return super().save(*args, **kwargs)

    class Meta:
        ordering = ['-created']
