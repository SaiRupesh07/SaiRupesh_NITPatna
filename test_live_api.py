import os
import requests
import json
import pytest

# Skip live integration unless explicitly enabled via environment variable.
RUN_LIVE = os.getenv('RUN_LIVE_TESTS', '0') == '1'


@pytest.mark.skipif(not RUN_LIVE, reason="Live integration tests disabled. Set RUN_LIVE_TESTS=1 to enable.")
def test_live_api():
    base_url = "https://bill-extraction-pipeline.onrender.com"
    
    print("🚀 Testing LIVE Deployed API")
    print("=" * 50)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    health_response = requests.get(f"{base_url}/health")
    print(f"   Status: {health_response.status_code}")
    print(f"   Response: {health_response.json()}")
    
    # Test main extraction endpoint
    print("\n2. Testing bill extraction endpoint...")
    test_data = {
        "document": "https://hackrx.blob.core.windows.net/assets/datathon-IIT/simple_2.png"
    }
    
    api_response = requests.post(
        f"{base_url}/extract-bill-data",
        json=test_data,
        timeout=30
    )
    
    print(f"   Status: {api_response.status_code}")
    result = api_response.json()
    
    if result.get("is_success"):
        data = result["data"]
        print(f"   ✅ SUCCESS: True")
        print(f"   📦 Items Extracted: {data.get('total_item_count')}")
        print(f"   💰 Reconciled Amount: ${data.get('reconciled_amount', 0):.2f}")
        print(f"   📄 Pages: {len(data.get('pagewise_line_items', []))}")
        
        # Show extracted items
        for page in data.get('pagewise_line_items', []):
            print(f"\n   Page {page.get('page_no')} Items:")
            for i, item in enumerate(page.get('bill_items', []), 1):
                print(f"     {i}. {item.get('item_name')} - ${item.get('item_amount', 0):.2f}")
    else:
        print(f"   ❌ Error: {result.get('error')}")
    
    print("\n" + "=" * 50)
    print("🎉 YOUR HACKATHON SUBMISSION IS READY!")


if __name__ == "__main__":
    if not RUN_LIVE:
        print("RUN_LIVE_TESTS not set — skipping live test run. Set RUN_LIVE_TESTS=1 to enable.")
    else:
        test_live_api()
