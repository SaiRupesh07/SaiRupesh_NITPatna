import requests
import json
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_api():
    """Test the API with a sample request"""
    
    # Sample test data
    test_data = {
        "document": "https://hackrx.blob.core.windows.net/assets/datathon-IIT/simple_2.png"
    }
    
    print("🧪 Testing Bill Extraction API...")
    print(f"📄 Request URL: {test_data['document']}")
    
    try:
        start_time = time.time()
        
        # Updated to port 8001
        response = requests.post(
            "http://localhost:8001/extract-bill-data",  # Changed to 8001
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        end_time = time.time()
        
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"⏱️ Response Time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['is_success']}")
            
            if result['is_success']:
                data = result['data']
                print(f"📦 Total Items: {data['total_item_count']}")
                print(f"💰 Reconciled Amount: ${data['reconciled_amount']:.2f}")
                
                for page in data['pagewise_line_items']:
                    print(f"\n📄 Page {page['page_no']}:")
                    for i, item in enumerate(page['bill_items'], 1):
                        print(f"  {i}. {item['item_name']}")
                        print(f"     Amount: ${item['item_amount']:.2f}")
                        print(f"     Rate: ${item['item_rate']:.2f}")
                        print(f"     Quantity: {item['item_quantity']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"📝 Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the server is running on http://localhost:8001")
    except requests.exceptions.Timeout:
        print("❌ Request Timeout: The server took too long to respond")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_health():
    """Test health endpoint"""
    try:
        # Updated to port 8001
        response = requests.get("http://localhost:8001/health", timeout=10)
        print(f"\n🏥 Health Check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

if __name__ == "__main__":
    print("🚀 Bill Extraction Pipeline Test Suite")
    print("=" * 50)
    
    # Test health endpoint first
    test_health()
    
    print("\n" + "=" * 50)
    
    # Test main extraction endpoint
    test_api()